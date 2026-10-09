"""Evaluation harness: recall@k, faithfulness, correctness, refusal.

Run it two ways::

    python -m eval.harness                          # offline, hashing embedder
    python -m eval.harness --json                   # machine-readable
    MINING_EMBEDDER=openai LLM_API_KEY=... python -m eval.harness

What it does
------------
Loads the persisted index, asks every question in :mod:`eval.ground_truth` through
the same :class:`~pipeline.answer.AnswerEngine` the API uses, and scores the result.
It never re-implements retrieval: measuring a *different* pipeline from the one that
serves would prove nothing about the one that serves.

Reproducibility
---------------
The corpus is deterministic (same date -> same documents), the embedder is
deterministic, and the tie-breaking in the store is total, so the reported metrics
are byte-stable across runs. ``latency_ms`` is the one field that is not, and it is
kept out of every aggregate -- it is diagnostic, not a score.

Reading the numbers
-------------------
``recall_at_k`` and ``mrr`` are retrieval properties. ``faithfulness`` is a property
of the prose; on the extractive path it is 1.0 by construction, so a dip there means
the *LLM* interpolated something. ``correctness`` is what separates "retrieved the
wrong document" from "quoted the right document badly": a question can be recalled,
faithfully quoted, and still wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.ground_truth import QUESTIONS, REFUSAL_CASES
from eval.metrics import (
    AnswerScore,
    MetricSummary,
    RecallScore,
    citation_validity,
    fact_coverage,
    hit_rank,
    summarise,
    support_ratio,
)
from pipeline.answer import AnswerEngine
from pipeline.config import get_settings
from pipeline.embedding import build_embedder, describe_embedder
from pipeline.ingest import load_index
from pipeline.llm import build_chat_client
from pipeline.logging_setup import configure_logging, get_logger
from pipeline.retrieval import Retriever

__all__ = ["EvalReport", "run_eval"]

_log = get_logger(__name__)


@dataclass(slots=True)
class EvalReport:
    """The whole run: the summary, the raw per-question scores, and the provenance."""

    summary: MetricSummary
    provenance: dict[str, Any] = field(default_factory=dict)
    refusal_detail: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.as_dict(),
            "provenance": self.provenance,
            "refusal_detail": self.refusal_detail,
        }


def run_eval(
    *,
    index_dir: Path | None = None,
    top_k: int = 5,
    synthesise: bool = True,
) -> EvalReport:
    """Score every question against the persisted index."""
    store, _ = load_index(index_dir=index_dir)
    embedder = build_embedder()
    chat = build_chat_client()
    engine = AnswerEngine(Retriever(store, embedder), chat)

    recalls: list[RecallScore] = []
    answers: list[AnswerScore] = []
    rows: list[dict[str, Any]] = []

    for question in QUESTIONS:
        result = engine.answer(question.question, top_k=top_k, synthesise=synthesise)
        retrieved_urls = [item.chunk.url for item in result.retrieved]
        evidence = [item.chunk.text for item in result.retrieved]

        rank = hit_rank(retrieved_urls, question.expected_urls)
        recalls.append(RecallScore(question.id, rank, top_k))
        score = AnswerScore(
            question_id=question.id,
            mode=result.mode,
            faithfulness=support_ratio(result.answer, evidence),
            correctness=fact_coverage(result.answer, question.key_facts),
            citation_validity=citation_validity(result.answer, len(result.citations)),
            answer_chars=len(result.answer),
            latency_ms=result.latency_ms,
            rank=rank,
        )
        answers.append(score)
        rows.append(
            {
                "id": question.id,
                "question": question.question,
                "reference": question.reference,
                "answer": result.answer,
                "rank": rank,
                "top_urls": retrieved_urls[:top_k],
                "expected_urls": list(question.expected_urls),
                **asdict(score),
            }
        )
        _log.info(
            "scored question",
            extra={
                "extra_fields": {
                    "id": question.id,
                    "rank": rank,
                    "faithfulness": round(score.faithfulness, 3),
                    "correctness": round(score.correctness, 3),
                }
            },
        )

    # Refusals are scored separately: recall@k is undefined when nothing should be
    # retrieved, and folding them in would let a system that refuses everything and
    # one that answers everything both score 0.5.
    refusal_ok = 0
    refusal_detail: list[dict[str, Any]] = []
    for case in REFUSAL_CASES:
        result = engine.answer(case.question, top_k=top_k, synthesise=synthesise)
        refused = result.mode == "refusal" or not result.retrieved
        refusal_ok += int(refused)
        refusal_detail.append(
            {
                "id": case.id,
                "question": case.question,
                "refused": refused,
                "mode": result.mode,
                "answer": result.answer[:240],
                "retrieved": len(result.retrieved),
                "notes": case.notes,
            }
        )

    refusal_accuracy = refusal_ok / len(REFUSAL_CASES) if REFUSAL_CASES else 0.0
    summary = summarise(recalls, answers, top_k=top_k, refusal_accuracy=refusal_accuracy)

    stats = store.stats()
    provenance = {
        "generated_at": datetime.now(UTC).isoformat(),
        "embedder": describe_embedder(embedder),
        "backend": store.backend,
        "documents": stats.documents,
        "chunks": stats.chunks,
        "per_kind": stats.per_kind,
        "answerer": "llm" if chat is not None else "extractive",
        "top_k": top_k,
        "questions": len(QUESTIONS),
        "per_question_raw": rows,
    }
    return EvalReport(summary=summary, provenance=provenance, refusal_detail=refusal_detail)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _render_markdown(report: EvalReport) -> str:
    summary = report.summary
    prov = report.provenance
    embedder = prov["embedder"]
    embedder_label = (
        f"{embedder.get('name')} ({embedder.get('dim')}d)"
        if isinstance(embedder, dict)
        else str(embedder)
    )
    lines = [
        "# Evaluation report",
        "",
        "> 各指标的含义与读法见仓库根目录的 `DATA_NOTES.md` §5。指标名称"
        "（`recall@5`、`MRR`、`faithfulness` …）是代码与 CI 中的标识符，刻意保留英文，"
        "以便与日志、`report.json` 逐字对应。",
        "",
        f"- generated: {prov['generated_at']}",
        f"- answerer: `{prov['answerer']}`  |  embedder: `{embedder_label}`",
        f"- index: {prov['documents']} documents / {prov['chunks']} chunks ({prov['backend']})",
        f"- questions: {prov['questions']}  |  top_k: {prov['top_k']}",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| recall@{summary.top_k} | **{summary.recall_at_k:.3f}** |",
        f"| MRR | {summary.mrr:.3f} |",
        f"| answer faithfulness | {summary.faithfulness:.3f} |",
        f"| answer correctness | {summary.correctness:.3f} |",
        f"| citation validity | {summary.citation_validity:.3f} |",
        f"| refusal accuracy | {summary.refusal_accuracy:.3f} |",
        "",
    ]
    if summary.misses:
        lines += [
            "### Recall misses",
            "",
            *[f"- `{miss}`" for miss in summary.misses],
            "",
        ]

    lines += [
        "## Per question",
        "",
        "| id | rank | faithful | correct | cited | mode | question |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    details = {row["id"]: row for row in prov["per_question_raw"]}
    for item in summary.per_question:
        row = details.get(item.question_id, {})
        lines.append(
            f"| `{item.question_id}` | {item.rank or '—'} | {item.faithfulness:.2f} | "
            f"{item.correctness:.2f} | {item.citation_validity:.2f} | {item.mode} | "
            f"{(row.get('question') or '')[:52]} |"
        )

    lines += ["", "## Refusals", ""]
    for case in report.refusal_detail:
        mark = "OK" if case["refused"] else "MISS"
        lines.append(
            f"- **{mark}** `{case['id']}` — {case['notes']} "
            f"(mode={case['mode']}, retrieved={case['retrieved']})"
        )
    lines.append("")
    return "\n".join(lines)


def _embedder_label(provenance: dict[str, Any]) -> str:
    """``hashing (512d)`` rather than the raw dict repr."""
    embedder = provenance.get("embedder")
    if isinstance(embedder, dict):
        return f"{embedder.get('name')} ({embedder.get('dim')}d)"
    return str(embedder)


def _print_console(report: EvalReport) -> None:
    summary = report.summary
    prov = report.provenance
    print()
    print(f"index      : {prov['documents']} docs / {prov['chunks']} chunks ({prov['backend']})")
    print(f"embedder   : {_embedder_label(prov)}")
    print(f"answerer   : {prov['answerer']}")
    print("-" * 58)
    hit_count = summary.n - len(summary.misses)
    print(f"recall@{summary.top_k}  : {summary.recall_at_k:.3f}   ({hit_count}/{summary.n})")
    print(f"MRR        : {summary.mrr:.3f}")
    print(f"faithful   : {summary.faithfulness:.3f}")
    print(f"correct    : {summary.correctness:.3f}")
    print(f"citations  : {summary.citation_validity:.3f}")
    print(f"refusals   : {summary.refusal_accuracy:.3f}")
    if summary.misses:
        print(f"misses     : {', '.join(summary.misses)}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.harness", description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--no-synthesise",
        action="store_true",
        help="Force the extractive answerer even when an LLM key is configured.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    parser.add_argument("--out", type=Path, default=Path("eval"), help="Report directory.")
    parser.add_argument(
        "--no-write", action="store_true", help="Print only; do not write report files."
    )
    args = parser.parse_args(argv)
    configure_logging(get_settings().model_copy(update={"log_level": "WARNING"}), force=True)

    report = run_eval(
        index_dir=args.index_dir,
        top_k=args.top_k,
        synthesise=not args.no_synthesise,
    )

    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_console(report)

    if not args.no_write:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.out / "report.md").write_text(_render_markdown(report), encoding="utf-8")
        _log.info("report written", extra={"extra_fields": {"dir": str(args.out)}})

    # Non-zero on a hard regression, so CI can gate on it. The thresholds are set
    # just under the measured offline baseline (recall 1.0, faithfulness 1.0) with
    # room for a legitimate refactor to move a single question.
    ok = report.summary.recall_at_k >= 0.9 and report.summary.faithfulness >= 0.95
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
