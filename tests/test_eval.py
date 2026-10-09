"""Evaluation metrics, and the integrity of the ground truth itself.

Ground-truth integrity gets its own tests because a harness whose expected answers
point at nothing will happily report a recall of zero forever, and that reads like a
retrieval bug rather than a broken test fixture.
"""

from __future__ import annotations

from eval.ground_truth import (
    PRICE_QUESTIONS,
    QUESTIONS,
    REFUSAL_CASES,
    Question,
    price_key_facts,
)
from eval.metrics import (
    AnswerScore,
    RecallScore,
    citation_indices,
    citation_validity,
    fact_coverage,
    hit_rank,
    recall_at_k,
    summarise,
    support_ratio,
)
from pipeline.render import FLAT_THRESHOLD_PCT
from pipeline.synthetic.anchors import ANCHORS, PRICE_TARGETS


class TestRecall:
    def test_hit_at_each_rank(self) -> None:
        urls = ["a", "b", "c", "d", "e"]
        assert hit_rank(urls, {"e"}) == 5
        assert hit_rank(urls, {"a"}) == 1
        assert hit_rank(urls, {"z"}) is None

    def test_any_expected_url_counts(self) -> None:
        assert hit_rank(["x", "b"], {"a", "b"}) == 2

    def test_recall_respects_k(self) -> None:
        urls = ["a", "b", "c", "d", "e"]
        assert recall_at_k(urls, {"e"}, 5) == 1.0
        assert recall_at_k(urls, {"e"}, 4) == 0.0

    def test_no_expected_url_is_not_a_hit(self) -> None:
        assert recall_at_k(["a"], set(), 5) == 0.0

    def test_empty_ranking_is_a_miss(self) -> None:
        assert recall_at_k([], {"a"}, 5) == 0.0


class TestFaithfulness:
    def test_a_verbatim_quote_is_supported(self) -> None:
        evidence = ["Zinc treatment charges fell to multi-year lows after two mine closures."]
        assert support_ratio(evidence[0], evidence) == 1.0

    def test_an_invented_claim_is_not_supported(self) -> None:
        evidence = ["Zinc treatment charges fell to multi-year lows after two mine closures."]
        answer = "Platinum supply is expected to tighten sharply through the next decade."
        assert support_ratio(answer, evidence) < 0.5

    def test_a_light_rewording_is_still_supported(self) -> None:
        evidence = ["Indonesia cut the 2027 nickel ore production quota to 240 million tonnes."]
        answer = "Indonesia cut the 2027 nickel ore production quota to 240 million tonnes [1]."
        assert support_ratio(answer, evidence) == 1.0

    def test_empty_answer_scores_one(self) -> None:
        # Refusing says nothing unsupported; emptiness is caught by correctness.
        assert support_ratio("", ["anything"]) == 1.0

    def test_short_fragments_are_ignored(self) -> None:
        assert support_ratio("2026。", ["unrelated evidence about zinc"]) == 1.0


class TestFactCoverage:
    def test_all_groups_satisfied(self) -> None:
        answer = "Quota cut to 135000 tonnes, a fall of 13%."
        assert fact_coverage(answer, (("135000",), ("13",))) == 1.0

    def test_any_alternative_satisfies_a_group(self) -> None:
        assert fact_coverage("价格下跌 18.50%", (("down", "下跌"),)) == 1.0

    def test_missing_fact_lowers_the_score(self) -> None:
        assert fact_coverage("only 135000", (("135000",), ("172000",))) == 0.5

    def test_no_facts_scores_one(self) -> None:
        assert fact_coverage("anything", ()) == 1.0

    def test_matching_ignores_formatting(self) -> None:
        assert fact_coverage("A$1,200", (("1,200",),)) == 1.0


class TestCitationValidity:
    def test_all_markers_in_range(self) -> None:
        assert citation_validity("Quota is 240 Mt [1] and fell [2].", 2) == 1.0

    def test_out_of_range_marker_is_invalid(self) -> None:
        assert citation_validity("See [7].", 5) == 0.0

    def test_partial_validity(self) -> None:
        assert citation_validity("See [1] and [9].", 5) == 0.5

    def test_uncited_answer_is_invalid_when_evidence_existed(self) -> None:
        assert citation_validity("No markers here.", 3) == 0.0

    def test_uncited_refusal_is_valid(self) -> None:
        assert citation_validity("The corpus does not cover this.", 0) == 1.0

    def test_indices_are_extracted_in_order(self) -> None:
        assert citation_indices("a [2] b [1] c [2]") == [2, 1, 2]


class TestSummarise:
    def _answer(self, qid: str, **kwargs) -> AnswerScore:
        defaults: dict = {
            "question_id": qid,
            "mode": "extractive",
            "faithfulness": 1.0,
            "correctness": 1.0,
            "citation_validity": 1.0,
            "answer_chars": 100,
            "latency_ms": 5,
            "rank": 1,
        }
        defaults.update(kwargs)
        return AnswerScore(**defaults)

    def test_recall_and_mrr(self) -> None:
        recalls = [RecallScore("a", 1, 5), RecallScore("b", 2, 5), RecallScore("c", None, 5)]
        summary = summarise(recalls, [self._answer("a")], top_k=5)
        assert summary.n == 3
        assert summary.recall_at_k == 2 / 3
        assert summary.mrr == (1.0 + 0.5) / 3

    def test_misses_are_named(self) -> None:
        recalls = [RecallScore("hit", 1, 5), RecallScore("miss", None, 5)]
        assert summarise(recalls, [], top_k=5).misses == ["miss"]

    def test_empty_input_does_not_divide_by_zero(self) -> None:
        summary = summarise([], [], top_k=5)
        assert summary.recall_at_k == 0.0
        assert summary.mrr == 0.0

    def test_as_dict_is_json_safe(self) -> None:
        import json

        summary = summarise([RecallScore("a", 1, 5)], [self._answer("a")], top_k=5)
        json.dumps(summary.as_dict())


class TestGroundTruthIntegrity:
    def test_exactly_twenty_graded_questions(self) -> None:
        assert len(QUESTIONS) == 20

    def test_question_ids_are_unique(self) -> None:
        ids = [q.id for q in QUESTIONS]
        assert len(ids) == len(set(ids))

    def test_every_anchor_question_resolves_to_a_real_url(self) -> None:
        known = {anchor.url for anchor in ANCHORS}
        for question in QUESTIONS:
            for url in question.expected_urls:
                if url.startswith("https://www.") or url.startswith("https://industry"):
                    assert url in known, f"{question.id} points at an unknown anchor"

    def test_anchor_coverage_includes_every_planned_family(self) -> None:
        kinds = {anchor.kind.value for anchor in ANCHORS}
        assert kinds == {"news", "policy"}

    def test_price_questions_point_at_the_summary_document(self) -> None:
        for question in PRICE_QUESTIONS:
            assert question.expected_urls
            assert all(url.endswith("/trend-60d") for url in question.expected_urls)

    def test_price_facts_match_the_targets(self) -> None:
        # Derived, not typed: if a target changes, the expectation follows.
        for commodity in PRICE_TARGETS:
            facts = price_key_facts(commodity)
            target, _ = PRICE_TARGETS[commodity]
            magnitude = facts[0][0]
            assert magnitude == f"{abs(target) * 100:.2f}"

    def test_flat_commodity_expects_flat(self) -> None:
        target, _ = PRICE_TARGETS["iron_ore"]
        assert abs(target) * 100 < FLAT_THRESHOLD_PCT
        assert "flat" in price_key_facts("iron_ore")[1]

    def test_rising_commodity_expects_up(self) -> None:
        assert "up" in price_key_facts("copper")[1]

    def test_every_question_has_a_reference(self) -> None:
        assert all(q.reference for q in QUESTIONS)

    def test_key_facts_are_never_empty_for_graded_questions(self) -> None:
        assert all(q.key_facts for q in QUESTIONS)

    def test_refusal_cases_expect_no_document(self) -> None:
        assert REFUSAL_CASES
        assert all(not q.expected_urls for q in REFUSAL_CASES)

    def test_total_set_size_is_reported(self) -> None:
        assert len(QUESTIONS) + len(REFUSAL_CASES) == 22


def test_question_is_immutable() -> None:
    question = Question(id="x", question="q", expected_urls=("u",))
    try:
        question.id = "y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Question should be frozen so the ground truth cannot drift at runtime")
