"""The HTTP surface: health, query, refusal, validation, reload, guarded ingest."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from pipeline.config import Settings
from serve.app import create_app


@pytest.fixture
def client(saved_index: Settings):
    with TestClient(create_app(saved_index)) as test_client:
        yield test_client


class TestHealth:
    def test_health_reports_a_loaded_index(self, client) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["index_loaded"] is True
        assert body["documents"] == 3
        assert body["answerer"] == "extractive"

    def test_health_is_degraded_when_the_index_is_missing(self, settings: Settings) -> None:
        # The service must start and say why, rather than crash-loop.
        with TestClient(create_app(settings)) as client:
            body = client.get("/health").json()
            assert body["status"] == "degraded"
            assert body["index_loaded"] is False
            assert "pipeline.ingest" in body["error"]

    def test_root_lists_the_endpoints(self, client) -> None:
        body = client.get("/").json()
        assert body["service"] == "mining-three-source-pipeline"
        assert "/query" in body["endpoints"]


class TestStats:
    def test_stats_counts_documents_per_kind(self, client) -> None:
        body = client.get("/stats").json()
        assert body["documents"] == 3
        assert body["per_kind"] == {"news": 2, "policy": 1}
        assert body["backend"] == "numpy"

    def test_stats_is_503_without_an_index(self, settings: Settings) -> None:
        with TestClient(create_app(settings)) as client:
            assert client.get("/stats").status_code == 503


class TestQuery:
    def test_returns_an_answer_with_citations(self, client) -> None:
        response = client.post(
            "/query",
            json={"question": "印度尼西亚 2027 年镍矿配额调整了多少？", "top_k": 3},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["citations"]
        assert body["retrieved"]
        assert body["mode"] in {"extractive", "llm"}

    def test_reports_the_filters_it_applied(self, client) -> None:
        body = client.post("/query", json={"question": "中国镍政策"}).json()
        assert body["filters"]["jurisdiction"] == "CN"

    def test_explicit_filters_are_honoured(self, client) -> None:
        body = client.post(
            "/query",
            json={"question": "镍的价格怎么样", "filters": {"commodity": "copper"}},
        ).json()
        assert body["filters"]["commodity"] == "copper"

    def test_uncovered_commodity_is_refused(self, client) -> None:
        body = client.post("/query", json={"question": "铂金价格走势如何？"}).json()
        assert body["mode"] == "refusal"
        assert body["citations"] == []

    def test_synthesise_false_is_accepted(self, client) -> None:
        response = client.post("/query", json={"question": "铜库存", "synthesise": False})
        assert response.status_code == 200
        assert response.json()["mode"] == "extractive"

    def test_query_is_503_without_an_index(self, settings: Settings) -> None:
        with TestClient(create_app(settings)) as client:
            response = client.post("/query", json={"question": "铜价"})
            assert response.status_code == 503


class TestValidation:
    def test_unknown_field_is_rejected(self, client) -> None:
        # "topk" must not be silently ignored and return five results.
        response = client.post("/query", json={"question": "铜价", "topk": 20})
        assert response.status_code == 422

    def test_too_short_question_is_rejected(self, client) -> None:
        assert client.post("/query", json={"question": "x"}).status_code == 422

    def test_top_k_out_of_range_is_rejected(self, client) -> None:
        assert client.post("/query", json={"question": "铜价", "top_k": 0}).status_code == 422

    def test_missing_question_is_rejected(self, client) -> None:
        assert client.post("/query", json={}).status_code == 422


class TestReload:
    def test_reload_rereads_the_index(self, client) -> None:
        body = client.post("/reload").json()
        assert body["reloaded"] is True
        assert body["documents"] == 3
        assert body["took_ms"] >= 0


class TestIngestEndpoint:
    def test_disabled_by_default(self, client) -> None:
        response = client.post("/ingest")
        assert response.status_code == 403
        assert "MINING_ALLOW_INGEST_ENDPOINT" in response.json()["detail"]

    def test_enabled_path_reports_the_run(self, saved_index: Settings, monkeypatch) -> None:
        from pipeline.ingest import IngestOutcome
        from pipeline.models import IngestStats

        fake = IngestOutcome(
            documents=[],
            chunks=[],
            stats=IngestStats(documents=7, chunks=9, per_kind={"news": 7}),
        )
        monkeypatch.setattr("serve.app.ingest", lambda *_a, **_k: fake)

        enabled = saved_index.model_copy(update={"allow_ingest_endpoint": True})
        with TestClient(create_app(enabled)) as client:
            body = client.post("/ingest").json()
        assert body["documents"] == 7
        assert body["chunks"] == 9

    def test_days_query_parameter_is_forwarded(self, saved_index: Settings, monkeypatch) -> None:
        from pipeline.ingest import IngestOutcome
        from pipeline.models import IngestStats

        seen: dict[str, Any] = {}

        # ``index_dir`` cannot be renamed: the endpoint passes it by keyword.
        def fake_ingest(_settings, *, days=None, index_dir=None):
            seen["days"] = days
            seen["index_dir"] = index_dir
            return IngestOutcome(stats=IngestStats())

        monkeypatch.setattr("serve.app.ingest", fake_ingest)
        enabled = saved_index.model_copy(update={"allow_ingest_endpoint": True})
        with TestClient(create_app(enabled)) as client:
            client.post("/ingest?days=14")
        assert seen["days"] == 14


def test_openapi_schema_is_served(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "/query" in schema["paths"]
    assert "QueryRequest" in schema["components"]["schemas"]
