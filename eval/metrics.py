"""Metrics for the evaluation harness: recall@k, faithfulness, correctness.

Deliberately dependency-free (standard library only) and deterministic. A metric
that itself needs a model to compute would make the harness's own output
unreproducible, which defeats the point of measuring.

**recall@k** -- did an expected document reach the top ``k`` chunks. This is a
retrieval property and says nothing about the prose.

**faithfulness** -- what fraction of the answer's sentences are *supported by the
retrieved evidence*. A sentence counts as supported when it is (a) quoted verbatim
from an evidence chunk, or (b) shares enough of its character trigrams with an
evidence sentence to be a rewording rather than an invention. The extractive
answerer scores 1.0 by construction -- every word is a quote -- so on that path the
metric is a regression guard, not a measurement. It becomes a real measurement on
the LLM path, which is exactly where a hallucinated figure would otherwise go
unnoticed.

**correctness** -- what fraction of the question's ``key_facts`` appear in the
answer. Faithful-but-wrong is a real failure mode: quoting the wrong document
correctly scores 1.0 on faithfulness and 0.0 here, and the two together are what
distinguish "the retriever failed" from "the answerer failed".

**citation validity** -- every ``[n]`` in the answer must address a real citation.
An answer that cites ``[7]`` when five chunks were retrieved is broken even if the
prose reads well.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from pipeline.textutils import normalize_for_match, split_sentences

__all__ = [
    "AnswerScore",
    "MetricSummary",
    "RecallScore",
    "citation_indices",
    "citation_validity",
    "fact_coverage",
    "hit_rank",
    "recall_at_k",
    "summarise",
    "support_ratio",
]

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")

#: A sentence shorter than this is not scored for support -- "2026 年 9 月 1 日。"
#: is a fragment that any evidence contains by coincidence.
_MIN_SUPPORT_CHARS = 6

#: Trigram overlap above which a sentence is treated as a rewording of evidence.
_SUPPORT_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def hit_rank(retrieved_urls: Sequence[str], expected_urls: Iterable[str]) -> int | None:
    """1-based rank of the first expected URL in the ranking, or ``None``.

    Rank of the *document* is taken as the rank of its best chunk, which is what a
    user sees: the citation list is ordered by chunk score.
    """
    wanted = set(expected_urls)
    if not wanted:
        return None
    for rank, url in enumerate(retrieved_urls, start=1):
        if url in wanted:
            return rank
    return None


def recall_at_k(retrieved_urls: Sequence[str], expected_urls: Iterable[str], k: int) -> float:
    """``1.0`` when any expected document is within the top ``k``, else ``0.0``."""
    rank = hit_rank(retrieved_urls, expected_urls)
    return 1.0 if rank is not None and rank <= k else 0.0


# ---------------------------------------------------------------------------
# Answer quality
# ---------------------------------------------------------------------------


def _trigrams(text: str) -> set[str]:
    normalized = normalize_for_match(text)
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[i : i + 3] for i in range(len(normalized) - 2)}


def _evidence_sentences(evidence: Sequence[str]) -> list[str]:
    sentences: list[str] = []
    for block in evidence:
        sentences.extend(split_sentences(block))
    return sentences


def support_ratio(answer: str, evidence: Sequence[str]) -> float:
    """Fraction of the answer's scoreable sentences that the evidence supports.

    Returns ``1.0`` for an empty answer (nothing unsupported was said) -- the
    emptiness is caught by :func:`fact_coverage` instead. Returning 0.0 here would
    punish a correct refusal, which is the opposite of what the harness is for.
    """
    sentences = [
        sentence
        for sentence in split_sentences(answer)
        if len(normalize_for_match(sentence)) >= _MIN_SUPPORT_CHARS
    ]
    if not sentences:
        return 1.0
    evidence_normalised = normalize_for_match("\n".join(evidence))
    evidence_sentences = _evidence_sentences(evidence)
    evidence_trigrams = [_trigrams(sentence) for sentence in evidence_sentences]

    supported = 0
    for sentence in sentences:
        normalized = normalize_for_match(sentence)
        if normalized and normalized in evidence_normalised:
            supported += 1
            continue
        grams = _trigrams(sentence)
        if not grams:
            supported += 1  # nothing checkable; do not count it against the answer
            continue
        best = max(
            (len(grams & other) / len(grams) for other in evidence_trigrams),
            default=0.0,
        )
        if best >= _SUPPORT_THRESHOLD:
            supported += 1
    return supported / len(sentences)


def fact_coverage(answer: str, key_facts: Sequence[tuple[str, ...]]) -> float:
    """Fraction of fact groups satisfied by the answer (any alternative passes)."""
    if not key_facts:
        return 1.0
    haystack = normalize_for_match(answer)
    satisfied = 0
    for group in key_facts:
        if any(normalize_for_match(option) in haystack for option in group):
            satisfied += 1
    return satisfied / len(key_facts)


def citation_indices(answer: str) -> list[int]:
    """Every ``[n]`` marker in the answer, in order of appearance."""
    return [int(match.group(1)) for match in _CITATION_RE.finditer(answer)]


def citation_validity(answer: str, n_citations: int) -> float:
    """Fraction of citation markers that address a real citation.

    An answer with no markers scores 1.0 only when there was nothing to cite (the
    refusal case); if evidence was retrieved, an uncited answer is not valid.
    """
    indices = citation_indices(answer)
    if not indices:
        return 1.0 if n_citations == 0 else 0.0
    valid = sum(1 for index in indices if 1 <= index <= n_citations)
    return valid / len(indices)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RecallScore:
    question_id: str
    rank: int | None
    top_k: int

    @property
    def hit(self) -> bool:
        return self.rank is not None and self.rank <= self.top_k


@dataclass(slots=True)
class AnswerScore:
    question_id: str
    mode: str
    faithfulness: float
    correctness: float
    citation_validity: float
    answer_chars: int
    latency_ms: int
    rank: int | None = None


@dataclass(slots=True)
class MetricSummary:
    """Aggregate numbers, plus the raw per-question scores they came from."""

    n: int = 0
    top_k: int = 5
    recall_at_k: float = 0.0
    mrr: float = 0.0
    faithfulness: float = 0.0
    correctness: float = 0.0
    citation_validity: float = 0.0
    refusal_accuracy: float = 0.0
    per_question: list[AnswerScore] = field(default_factory=list)
    misses: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "top_k": self.top_k,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "faithfulness": round(self.faithfulness, 4),
            "correctness": round(self.correctness, 4),
            "citation_validity": round(self.citation_validity, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "misses": self.misses,
            "per_question": [
                {
                    "id": item.question_id,
                    "mode": item.mode,
                    "rank": item.rank,
                    "faithfulness": round(item.faithfulness, 4),
                    "correctness": round(item.correctness, 4),
                    "citation_validity": round(item.citation_validity, 4),
                    "answer_chars": item.answer_chars,
                    "latency_ms": item.latency_ms,
                }
                for item in self.per_question
            ],
        }


def summarise(
    recalls: Sequence[RecallScore],
    answers: Sequence[AnswerScore],
    *,
    top_k: int,
    refusal_accuracy: float = 0.0,
) -> MetricSummary:
    """Roll per-question scores into one summary.

    MRR uses the same expected-URL match as recall: ``1/rank`` of the first hit, or
    0 when the document was not retrieved at all (rank ``None`` is *unretrieved*,
    which is different from retrieved-and-late only in the top-k cut).
    """

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    n_recall = len(recalls)
    return MetricSummary(
        n=n_recall,
        top_k=top_k,
        recall_at_k=(sum(1.0 for score in recalls if score.hit) / n_recall if n_recall else 0.0),
        mrr=mean([1.0 / score.rank if score.rank else 0.0 for score in recalls]),
        faithfulness=mean([item.faithfulness for item in answers]),
        correctness=mean([item.correctness for item in answers]),
        citation_validity=mean([item.citation_validity for item in answers]),
        refusal_accuracy=refusal_accuracy,
        per_question=list(answers),
        misses=[score.question_id for score in recalls if not score.hit],
    )
