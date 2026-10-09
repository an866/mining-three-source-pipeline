"""Query planning, coverage checks, window widening and hit de-duplication."""

from __future__ import annotations

from tests.conftest import index_documents, make_document

from pipeline.models import SourceKind
from pipeline.retrieval import (
    Retriever,
    dedupe_hits,
    mentioned_commodities,
    plan_filters,
    uncovered_minerals,
)


class TestPlannerGenreInference:
    def test_explicit_genre_words_set_the_filter(self) -> None:
        assert SourceKind.POLICY in plan_filters("近 7 天的政策有哪些变化").kinds
        assert SourceKind.PRICE in plan_filters("铜价行情如何").kinds

    def test_direction_words_do_not_imply_the_price_series(self) -> None:
        # The bug this guards: "锌加工费为何跌至多年低点" planned kind=price because of
        # the word 跌, and the price-only filter excluded the news story that answers it.
        filters = plan_filters("锌加工费为何跌至多年低点？")
        assert SourceKind.PRICE not in filters.kinds

    def test_a_bare_price_word_does_not_imply_the_series(self) -> None:
        # 价 is a substring of 价格/报价/油价; it is too generic to mean "give me the
        # daily close series".
        assert SourceKind.PRICE not in plan_filters("锂价").kinds

    def test_topic_word_quota_does_not_imply_policy(self) -> None:
        # Another measured miss: "Indonesia nickel ore quota 2027" is about a quota,
        # not a request for policy documents.
        filters = plan_filters("Indonesia nickel ore quota 2027")
        assert SourceKind.POLICY not in filters.kinds

    def test_no_genre_words_means_no_kind_filter(self) -> None:
        assert plan_filters("印度尼西亚 2027 年镍矿配额调整了多少？").kinds == []


class TestPlannerJurisdictionAndCommodity:
    def test_china_and_australia_are_recognised(self) -> None:
        assert plan_filters("中国稀土政策").jurisdiction == "CN"
        assert plan_filters("澳大利亚关键矿产清单").jurisdiction == "AU"

    def test_absent_jurisdiction_is_empty_not_a_default(self) -> None:
        assert plan_filters("印度尼西亚镍矿配额").jurisdiction == ""

    def test_longest_commodity_form_wins(self) -> None:
        # 铁矿石 must plan as iron_ore, not as 石墨 or 铝.
        assert plan_filters("铁矿石港口库存").commodity == "iron_ore"
        assert plan_filters("氧化铝价格").commodity == "alumina"

    def test_all_corpus_commodities_are_plannable(self) -> None:
        for question, expected in [
            ("碳酸锂价格", "lithium"),
            ("镓出口管制", "gallium"),
            ("锗出口管制", "germanium"),
            ("石墨出口", "graphite"),
            ("钼价格", "molybdenum"),
        ]:
            assert plan_filters(question).commodity == expected, question

    def test_mentioned_commodities_can_be_plural(self) -> None:
        assert set(mentioned_commodities("铜和锌的价格")) == {"copper", "zinc"}


class TestRecencyExtraction:
    def test_english_windows(self) -> None:
        assert plan_filters("copper prices over the last 7 days").days == 7
        assert plan_filters("last 2 weeks of news").days == 14
        assert plan_filters("past 3 months").days == 90

    def test_chinese_windows_with_digits_and_numerals_agree(self) -> None:
        assert plan_filters("近 30 天铜价").days == 30
        assert plan_filters("近一个月铜价").days == 30
        assert plan_filters("最近两周的新闻").days == 14

    def test_implicit_windows(self) -> None:
        assert plan_filters("今天的新闻").days == 1
        assert plan_filters("本周政策").days == 7

    def test_no_window_is_none_not_a_default(self) -> None:
        assert plan_filters("铜价走势").days is None

    def test_window_is_clamped_to_a_sane_range(self) -> None:
        assert plan_filters("近 9999 天").days == 3650


class TestCoverageCheck:
    def test_a_covered_commodity_is_not_flagged(self) -> None:
        assert uncovered_minerals("印尼镍矿配额", {"nickel", "copper"}) == []

    def test_a_known_but_absent_commodity_is_flagged(self) -> None:
        # Prices failed to collect this run: a price question must refuse rather than
        # answer from news.
        assert uncovered_minerals("镍价走势", {"copper"}) == ["nickel"]

    def test_an_out_of_corpus_mineral_is_flagged(self) -> None:
        assert uncovered_minerals("铂金价格走势如何？", {"nickel"}) == ["platinum"]
        assert uncovered_minerals("铀矿供应紧张吗？", {"nickel"}) == ["uranium"]

    def test_alumina_is_not_mistaken_for_aluminium(self) -> None:
        # 铝 is a substring of 氧化铝, which the corpus does cover. A naive substring
        # rule would refuse an answerable question.
        assert uncovered_minerals("氧化铝价格", {"alumina"}) == []

    def test_a_generic_question_flags_nothing(self) -> None:
        assert uncovered_minerals("最近有哪些矿业新闻？", {"nickel"}) == []


class TestWindowWidening:
    def _retriever(self, settings):
        # 铜价 plans commodity=copper, so the documents have to carry the commodity
        # or the filter excludes them and the test measures nothing.
        docs = [
            make_document(
                doc_id="fresh",
                title="Fresh copper",
                text="copper " * 60,
                age_days=5,
                commodity="copper",
            ),
            make_document(
                doc_id="older",
                title="Older copper",
                text="copper " * 60,
                age_days=40,
                commodity="copper",
            ),
        ]
        store, embedder = index_documents(docs, settings)
        return Retriever(store, embedder)

    def test_a_quiet_window_is_widened_rather_than_returning_nothing(self, settings) -> None:
        # 2 days is empty (newest is 5 days old); widening x4 reaches it.
        retriever = self._retriever(settings)
        assert retriever.retrieve("近 2 天铜价", top_k=5, widen=False) == []
        hits = retriever.retrieve("近 2 天铜价", top_k=5)
        assert {h.chunk.document_id for h in hits} == {"fresh"}

    def test_a_populated_window_is_not_widened(self, settings) -> None:
        retriever = self._retriever(settings)
        hits = retriever.retrieve("近 30 天铜价", top_k=5)
        assert {h.chunk.document_id for h in hits} == {"fresh"}

    def test_widening_can_be_disabled(self, settings) -> None:
        retriever = self._retriever(settings)
        assert retriever.retrieve("近 2 天铜价", top_k=5, widen=False) == []

    def test_jurisdiction_is_not_relaxed_by_widening(self, settings) -> None:
        docs = [
            make_document(
                doc_id="cn",
                title="中国铜政策",
                text="铜相关。" * 40,
                kind=SourceKind.POLICY,
                jurisdiction="CN",
                commodity="copper",
            ),
            make_document(
                doc_id="au",
                title="Australian copper",
                text="copper " * 40,
                kind=SourceKind.POLICY,
                jurisdiction="AU",
                commodity="copper",
            ),
        ]
        store, embedder = index_documents(docs, settings)
        hits = Retriever(store, embedder).retrieve("中国铜政策", top_k=5)
        assert {h.chunk.document_id for h in hits} == {"cn"}


class TestDedupeHits:
    def test_caps_representation_per_document(self, tiny_store) -> None:
        store, embedder = tiny_store
        hits = store.search(embedder.embed_query("copper"), query_text="copper", top_k=10)
        kept = dedupe_hits(hits, max_per_document=1)
        documents = [hit.chunk.document_id for hit in kept]
        assert len(documents) == len(set(documents))

    def test_keeps_the_first_two_of_a_dominant_document(self, settings) -> None:
        doc = make_document(
            doc_id="long", title="Long copper", text="copper inventory data. " * 400
        )
        store, embedder = index_documents([doc], settings)
        hits = store.search(embedder.embed_query("copper"), query_text="copper", top_k=10)
        assert len(hits) > 2
        assert len(dedupe_hits(hits, max_per_document=2)) == 2

    def test_order_is_preserved(self, tiny_store) -> None:
        store, embedder = tiny_store
        hits = store.search(
            embedder.embed_query("copper nickel"), query_text="copper nickel", top_k=10
        )
        kept = dedupe_hits(hits)
        assert [h.chunk.id for h in kept] == [h.chunk.id for h in hits if h in kept]
