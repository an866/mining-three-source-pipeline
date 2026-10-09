"""Answer synthesis: extraction, citations, the prompt, and coverage refusal."""

from __future__ import annotations

import pytest
from tests.conftest import index_documents, make_document

from pipeline.answer import (
    NO_EVIDENCE_ANSWER,
    SYSTEM_PROMPT,
    AnswerEngine,
    build_citations,
    build_prompt,
    coverage_refusal,
    extractive_answer,
)
from pipeline.errors import LLMError
from pipeline.retrieval import Retriever
from pipeline.vectorstore import SearchHit


def _hits(store, embedder, question: str, k: int = 5) -> list[SearchHit]:
    return store.search(embedder.embed_query(question), query_text=question, top_k=k)


class TestExtractiveAnswer:
    def test_quotes_the_evidence_with_rank_markers(self, tiny_store) -> None:
        store, embedder = tiny_store
        answer = extractive_answer(
            "印度尼西亚 2027 年镍矿配额调整了多少？", _hits(store, embedder, "镍 quota")
        )
        assert "[1]" in answer
        # Every sentence of an extractive answer is a quote, which is what makes
        # faithfulness 1.0 by construction.
        assert "Indonesia" in answer or "quota" in answer

    def test_no_hits_returns_the_refusal_not_an_empty_string(self) -> None:
        assert extractive_answer("anything", []) == NO_EVIDENCE_ANSWER

    def test_fragments_are_dropped(self, tiny_store) -> None:
        store, embedder = tiny_store
        answer = extractive_answer("铜库存", _hits(store, embedder, "copper"))
        # No single-word fragments dressed up as sentences.
        for part in answer.replace("[1]", "").replace("[2]", "").split():
            assert part

    def test_answer_is_bounded(self, settings) -> None:
        docs = [make_document(doc_id="d", title="Copper", text="copper inventory fell. " * 300)]
        store, embedder = index_documents(docs, settings)
        answer = extractive_answer("copper inventory", _hits(store, embedder, "copper"))
        assert len(answer) <= 1400

    def test_deterministic_for_the_same_input(self, tiny_store) -> None:
        store, embedder = tiny_store
        hits = _hits(store, embedder, "copper")
        assert extractive_answer("copper", hits) == extractive_answer("copper", hits)


class TestCitations:
    def test_numbering_starts_at_one_in_ranking_order(self, tiny_store) -> None:
        store, embedder = tiny_store
        hits = _hits(store, embedder, "copper nickel")
        citations = build_citations(hits)
        assert [c.index for c in citations] == list(range(1, len(hits) + 1))

    def test_citations_carry_a_url(self, tiny_store) -> None:
        store, embedder = tiny_store
        assert all(c.url for c in build_citations(_hits(store, embedder, "copper")))

    def test_empty_hits_produce_no_citations(self) -> None:
        assert build_citations([]) == []


class TestPrompt:
    def test_prompt_numbers_the_evidence(self, tiny_store) -> None:
        store, embedder = tiny_store
        messages = build_prompt("copper?", _hits(store, embedder, "copper"))
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert "[1]" in messages[1]["content"]
        assert messages[1]["content"].endswith("copper?")

    def test_prompt_says_so_when_there_is_no_evidence(self) -> None:
        messages = build_prompt("copper?", [])
        assert "no evidence" in messages[1]["content"].lower()

    def test_system_prompt_requires_refusal(self) -> None:
        # Faithfulness over 20 questions is uninformative if the model may
        # interpolate; the refusal clause is what stops it.
        assert "never fill the gap" in SYSTEM_PROMPT.lower()
        assert "only" in SYSTEM_PROMPT.lower()


class TestCoverageRefusal:
    def test_names_the_missing_commodity(self) -> None:
        message = coverage_refusal(["platinum"])
        assert "platinum" in message
        assert "铂金" in message

    def test_states_what_the_corpus_does_hold(self) -> None:
        message = coverage_refusal(["uranium"])
        assert "copper" in message
        assert "critical-minerals" in message


class TestAnswerEngine:
    def _engine(self, settings, chat=None) -> AnswerEngine:
        docs = [
            make_document(
                doc_id="nickel-id",
                title="Indonesia trims nickel ore quota for 2027",
                text="Indonesia cut the 2027 nickel ore quota to 240 million wet metric tonnes.",
                commodity="nickel",
                url="https://example.test/nickel",
            ),
            make_document(
                doc_id="copper",
                title="LME copper inventories fall",
                text="Copper inventories fell to a six-week low.",
                commodity="copper",
                url="https://example.test/copper",
            ),
        ]
        store, embedder = index_documents(docs, settings)
        return AnswerEngine(Retriever(store, embedder), chat, settings=settings)

    def test_answered_question_returns_citations(self, settings) -> None:
        result = self._engine(settings).answer("印度尼西亚 2027 年镍矿配额调整了多少？")
        assert result.citations
        assert result.retrieved
        assert result.mode == "extractive"

    def test_uncovered_commodity_is_refused_without_retrieval(self, settings) -> None:
        result = self._engine(settings).answer("铂金最近的价格走势如何？")
        assert result.mode == "refusal"
        assert result.retrieved == []
        assert result.citations == []
        assert "platinum" in result.answer

    def test_echoes_the_filters_actually_applied(self, settings) -> None:
        result = self._engine(settings).answer("中国镍政策")
        assert result.filters.jurisdiction == "CN"

    def test_explicit_filters_override_the_planner(self, settings) -> None:
        from pipeline.models import QueryFilters

        result = self._engine(settings).answer("镍", filters=QueryFilters(commodity="copper"))
        assert result.filters.commodity == "copper"

    def test_llm_failure_degrades_to_extractive(self, settings) -> None:
        class Broken:
            def complete(self, messages, **kwargs):
                raise LLMError("upstream 503")

        result = self._engine(settings, chat=Broken()).answer("印尼镍矿配额")
        assert result.mode == "extractive"
        assert result.answer

    def test_synthesise_false_skips_the_llm(self, settings) -> None:
        class Recording:
            called = False

            def complete(self, messages, **kwargs):
                self.called = True
                return "should not be used"

        chat = Recording()
        result = self._engine(settings, chat=chat).answer("印尼镍矿配额", synthesise=False)
        assert chat.called is False
        assert result.mode == "extractive"

    def test_llm_path_is_labelled(self, settings) -> None:
        class Echo:
            def complete(self, messages, **kwargs):
                return "Quota cut to 240 Mt [1]."

        result = self._engine(settings, chat=Echo()).answer("印尼镍矿配额")
        assert result.mode == "llm"
        assert result.answer == "Quota cut to 240 Mt [1]."

    def test_latency_is_recorded(self, settings) -> None:
        assert self._engine(settings).answer("铜库存").latency_ms >= 0

    def test_top_k_is_honoured(self, settings) -> None:
        result = self._engine(settings).answer("镍 铜", top_k=1)
        assert len(result.retrieved) <= 1


@pytest.mark.parametrize("question", ["", " "])
def test_engine_handles_degenerate_questions(settings, question: str) -> None:
    result = AnswerEngine(
        Retriever(
            *index_documents(
                [make_document(doc_id="d", title="Copper", text="copper " * 40)], settings
            )
        ),
        None,
        settings=settings,
    ).answer(question)
    assert result.answer
