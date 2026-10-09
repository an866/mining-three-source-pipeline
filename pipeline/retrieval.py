"""Query planning and retrieval.

Two responsibilities, deliberately separated from answering:

**Planning.** A question like "澳大利亚最近有哪些关键矿产政策变化" carries three
constraints a naive vector search throws away: a jurisdiction (AU), a source
family (policy) and a recency window (unbounded, but "最近" implies "prefer
recent"). :func:`plan_filters` extracts them with an explicit, auditable rule set
rather than a learned intent classifier -- there are eight rules, they are
readable, and when one is wrong it is a one-line fix. A model call here would add
latency to every query and make a wrong filter impossible to explain.

**Retrieval.** Delegated to the store, with one addition: the recency constraint is
applied as a *soft* preference in addition to the hard filter. A hard "近 7 天" cut
can return nothing when the window is genuinely quiet, and an empty answer is worse
than an answer from nine days ago that says how old it is. So the filtered search
runs first; if it under-fills, the window is widened and the citations carry their
real dates.

The planner's output is echoed in the API response, so a user can see exactly which
constraints were applied and why a result set looks the way it does.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

from pipeline.embedding import Embedder
from pipeline.logging_setup import get_logger
from pipeline.models import QueryFilters, SourceKind
from pipeline.vectorstore import SearchHit, VectorStore

__all__ = [
    "Retriever",
    "mentioned_commodities",
    "plan_filters",
    "uncovered_minerals",
]

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Query planning
# ---------------------------------------------------------------------------

_DAYS_RE = re.compile(
    r"(?:last|past|previous|近|最近|过去)\s*(\d{1,4})\s*(?:个)?\s*(?:day|days|天|日|交易日)",
    re.IGNORECASE,
)
_WEEKS_RE = re.compile(r"(?:last|past|近|最近|过去)\s*(\d{1,2})\s*(?:个)?\s*(?:week|weeks|周|星期)")
_MONTHS_RE = re.compile(r"(?:last|past|近|最近|过去)\s*(\d{1,2})\s*(?:个)?\s*(?:month|months|月)")

_IMPLICIT_WINDOW: tuple[tuple[str, int], ...] = (
    ("today", 1),
    ("今天", 1),
    ("昨日", 2),
    ("昨天", 2),
    ("this week", 7),
    ("本周", 7),
    ("这周", 7),
    ("this month", 30),
    ("本月", 30),
    ("这个月", 30),
    ("this quarter", 90),
    ("本季度", 90),
    ("this year", 365),
    ("今年", 365),
)

#: ``keyword -> source family``. A question can legitimately match more than one
#: ("价格新闻").
#:
#: Only words that name a *source family* belong here. An earlier version listed
#: "quota", which meant "Indonesia nickel ore quota 2027" planned a hard
#: ``kind=policy`` filter and silently excluded the news article that answered it --
#: the question was about a quota, not about wanting policy documents. Topic words
#: are topics; the planner only filters on genre.
#:
#: The same trap caught the price list. It carried "涨", "跌" and a bare "价", so
#: "锌加工费为何跌至多年低点" planned ``kind=price`` -- on the strength of the word
#: "跌" -- and hard-excluded the news story that answered it. Direction words say
#: which way a price moved, not that the user wants the daily price series. The list
#: now holds only terms that actually name the series or a way of quoting it
#: ("价格", "行情", "走势", "收盘", "spot", "futures").
_KIND_KEYWORDS: tuple[tuple[tuple[str, ...], SourceKind], ...] = (
    (
        (
            "policy",
            "policies",
            "regulation",
            "regulations",
            "regulatory",
            "notice",
            "announcement",
            "communique",
            "directive",
            "standard",
            "consultation",
            "catalogue",
            "strategy",
            "plan",
            "政策",
            "法规",
            "监管",
            "公告",
            "通知",
            "指标文件",
            "管制",
            "清单",
            "标准",
            "规划",
            "征求意见",
            "收储",
            "备案",
        ),
        SourceKind.POLICY,
    ),
    (
        (
            "price",
            "prices",
            "quotation",
            "quote",
            "market price",
            "trend",
            "trends",
            "settlement",
            "chart",
            "spot",
            "futures",
            "价格",
            "报价",
            "行情",
            "走势",
            "趋势",
            "收盘",
            "现货",
            "期货",
            "结算",
        ),
        SourceKind.PRICE,
    ),
    (
        (
            "news",
            "article",
            "articles",
            "report",
            "reports",
            "headline",
            "coverage",
            "新闻",
            "报道",
            "消息",
            "资讯",
            "快讯",
        ),
        SourceKind.NEWS,
    ),
)

_JURISDICTIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("china", "chinese", "prc", "中国", "国内", "我国", "工信部", "商务部", "稀土集团"), "CN"),
    (("australia", "australian", "disr", "澳大利亚", "澳洲", "西澳"), "AU"),
)

#: Commodity names -> the string stored on a document's ``commodity`` field.
#:
#: This is wider than the five priced benchmarks on purpose. Price *series* exist for
#: five commodities, but news and policy documents carry the full mineral set, so a
#: question about 镓 or 石墨 can still be scoped usefully. Order matters: the longest
#: and most specific forms come first, so "铁矿石" is not planned as "石墨".
_COMMODITIES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("iron ore", "62% fe", "65% fe", "铁矿石"), "iron_ore"),
    (("rare earth", "rare-earth", "neodymium", "praseodymium", "稀土"), "rare_earth"),
    (("lithium", "carbonate", "spodumene", "hydroxide", "锂", "碳酸锂", "氢氧化锂"), "lithium"),
    (("graphite", "石墨"), "graphite"),
    (("tungsten", "钨"), "tungsten"),
    (("manganese", "锰"), "manganese"),
    (("copper", "铜"), "copper"),
    (("zinc", "锌"), "zinc"),
    (("nickel", "镍"), "nickel"),
    (("cobalt", "钴"), "cobalt"),
    (("gallium", "镓"), "gallium"),
    (("germanium", "锗"), "germanium"),
    (("molybdenum", "钼"), "molybdenum"),
    (("alumina", "氧化铝"), "alumina"),
)


#: Minerals an analyst may name that this corpus carries **no** documents for.
#:
#: This powers a coverage check, not a filter. Naming one is a request the index
#: cannot satisfy, and the honest response is to say so rather than to return the
#: nearest note about some other metal. Measured on the probe set, "铂金（platinum）
#: 最近的价格走势如何？" scored 0.46 against the price series -- *above* several
#: genuinely answerable questions -- because it shares 价格/走势 with every price
#: document. A score threshold cannot separate those cases; a coverage check can.
#:
#: Bare "铝" is deliberately absent: it is a substring of 氧化铝 (alumina), which the
#: corpus does cover, so a substring rule would refuse a question it could answer.
_UNCOVERED_MINERALS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("platinum", "铂金", "铂"), "platinum"),
    (("uranium", "铀"), "uranium"),
    (("silver", "白银"), "silver"),
    (("gold", "黄金"), "gold"),
    (("tin", "锡"), "tin"),
    (("lead", "铅"), "lead"),
    (("aluminium", "aluminum"), "aluminium"),
    (("potash", "钾肥"), "potash"),
    (("antimony", "锑"), "antimony"),
    (("titanium", "钛"), "titanium"),
    (("vanadium", "钒"), "vanadium"),
)


def mentioned_commodities(question: str) -> list[str]:
    """Every commodity in the planning vocabulary that ``question`` names."""
    lowered = (question or "").lower()
    return [
        name for keywords, name in _COMMODITIES if any(keyword in lowered for keyword in keywords)
    ]


def uncovered_minerals(question: str, available: Collection[str]) -> list[str]:
    """Minerals named by ``question`` that no indexed document covers.

    Two ways to be uncovered: name a commodity the planner knows but the index has
    no documents for, or name one of the known-absent minerals. ``available`` is the
    set of commodities actually present in the index, so the answer stays correct
    after a partial ingest (a run where prices failed should refuse a price
    question, not answer it from news).
    """
    lowered = (question or "").lower()
    present = set(available)
    absent = {name for name in mentioned_commodities(question) if name not in present}
    outside = {
        name
        for keywords, name in _UNCOVERED_MINERALS
        if any(keyword in lowered for keyword in keywords)
    }
    return sorted(absent | outside)


def plan_filters(question: str) -> QueryFilters:
    """Extract structured constraints from a natural-language question.

    Returns an empty filter object when nothing is recognised, which callers treat
    as "search everything" rather than "match nothing".
    """
    text = question or ""
    lowered = text.lower()

    days = _extract_days(lowered)
    kinds = [
        kind for keywords, kind in _KIND_KEYWORDS if any(keyword in lowered for keyword in keywords)
    ]
    jurisdiction = next(
        (
            code
            for keywords, code in _JURISDICTIONS
            if any(keyword in lowered for keyword in keywords)
        ),
        "",
    )
    commodity = next(
        (
            name
            for keywords, name in _COMMODITIES
            if any(keyword in lowered for keyword in keywords)
        ),
        "",
    )

    # "近 30 天的铜价" names a commodity and a window; a bare "锂价怎么走" names only
    # the commodity. Both are fine -- an unset field means "unconstrained".
    return QueryFilters(
        kinds=kinds,
        days=days,
        jurisdiction=jurisdiction,
        commodity=commodity,
    )


def _extract_days(lowered: str) -> int | None:
    """Recency window in days, or ``None`` when the question does not set one.

    Chinese numerals are transliterated first so that "近一个月" and "近 1 个月"
    plan identically -- a user writing the former and getting an unbounded search
    while the latter is bounded would be an inexplicable inconsistency.
    """
    text = _normalise_cn_numerals(lowered)
    if match := _DAYS_RE.search(text):
        return max(1, min(3650, int(match.group(1))))
    if match := _WEEKS_RE.search(text):
        return max(1, min(3650, int(match.group(1)) * 7))
    if match := _MONTHS_RE.search(text):
        return max(1, min(3650, int(match.group(1)) * 30))
    for needle, window in _IMPLICIT_WINDOW:
        if needle in lowered:
            return window
    return None


_CN_DIGITS: dict[str, str] = {
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}
_CN_NUMERAL_RE = re.compile(r"[一二两三四五六七八九十]+")


def _normalise_cn_numerals(text: str) -> str:
    """Rewrite ``一``/``十二``/``三十`` as digits, leaving other text alone."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if "十" in token:
            tens, _, ones = token.partition("十")
            high = int(_CN_DIGITS.get(tens, "1")) if tens else 1
            low = int(_CN_DIGITS.get(ones, "0")) if ones else 0
            return str(high * 10 + low)
        return "".join(_CN_DIGITS.get(char, char) for char in token)

    return _CN_NUMERAL_RE.sub(replace, text)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

#: When a hard window under-fills, retry with the window multiplied by this.
_WIDEN_FACTOR = 4


class Retriever:
    """Ties an embedder to a vector store and adds window-widening."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        filters: QueryFilters | None = None,
        lexical_weight: float | None = None,
        widen: bool = True,
    ) -> list[SearchHit]:
        """Best ``top_k`` chunks for ``question``.

        Window-widening only ever relaxes the *recency* constraint. Source family
        stays hard: "近 7 天的政策" is an explicit request for policy documents, and
        returning a price row instead would be answering a different question.

        Jurisdiction and commodity stay hard too, but not against every source. The
        filter is applied where the field is meaningful -- a jurisdiction only
        excludes a chunk that has one -- so naming a country cannot silently delete
        the wire copy that reports on it. See :meth:`NumpyVectorStore._mask`.
        """
        planned = filters if filters is not None else plan_filters(question)
        vector = self.embedder.embed_query(question)
        hits = self.store.search(
            vector,
            query_text=question,
            top_k=top_k,
            filters=planned,
            lexical_weight=lexical_weight,
        )
        if hits or not widen or not planned.days:
            return hits

        widened = planned.model_copy(update={"days": planned.days * _WIDEN_FACTOR})
        loose = self.store.search(
            vector,
            query_text=question,
            top_k=top_k,
            filters=widened,
            lexical_weight=lexical_weight,
        )
        if loose:
            _log.info(
                "widened recency window to fill the result set",
                extra={
                    "extra_fields": {
                        "requested_days": planned.days,
                        "widened_days": widened.days,
                        "hits": len(loose),
                    }
                },
            )
        return loose or hits


def dedupe_hits(hits: Sequence[SearchHit], *, max_per_document: int = 2) -> list[SearchHit]:
    """Cap how many chunks of one document reach the answer.

    Without this, a single long policy PDF can occupy all five slots and the
    answer is built from one source. ``max_per_document=2`` keeps multi-source
    evidence in the window while still allowing the two best spans of a document
    that is genuinely the best source.
    """
    counts: dict[str, int] = {}
    kept: list[SearchHit] = []
    for hit in hits:
        used = counts.get(hit.chunk.document_id, 0)
        if used >= max_per_document:
            continue
        counts[hit.chunk.document_id] = used + 1
        kept.append(hit)
    return kept
