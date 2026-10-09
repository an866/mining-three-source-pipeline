"""Text utilities: normalisation, de-duplication, chunking and lexical scoring.

Three problems this module exists to solve.

**De-duplication.** Mining news is syndicated aggressively -- the same story
appears on mining.com, in a Google News cluster and in a company release -- so a
URL set alone is not enough. We use three tiers, described in
``dedupe_fingerprint``/:func:`dedupe_key``.

**Bilingual safety.** Half the policy corpus is Chinese (中国稀土集团, 工信部) and
half is English (DISR). Anything that normalises by stripping non-ASCII silently
deletes the entire Chinese corpus, so normalisation here is Unicode-native and
tokenisation emits CJK bigrams rather than treating 碳酸锂 as one opaque token.

**Chunking with real offsets.** Retrieval has to cite a span, not a whole
document. :func:`chunk_text` returns character offsets into the original text so
a citation can be re-read in place.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

__all__ = [
    "article_id",
    "chunk_text",
    "concepts_in",
    "content_hash",
    "dedupe_fingerprint",
    "dedupe_key",
    "document_id",
    "has_cjk",
    "jaccard",
    "lexical_score",
    "normalize_for_match",
    "normalize_title",
    "normalize_whitespace",
    "parse_datetime",
    "relevance_score",
    "search_tokens",
    "search_units",
    "shingles",
    "slugify",
    "split_sentences",
    "strip_tracking",
    "tokenize",
    "truncate",
]

_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "src",
        "spm",
        "from",
        "share_token",
    }
)

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)

#: CJK ideographs, kana and compatibility ideographs. Deliberately excludes the
#: CJK punctuation range, which the sentence splitter handles instead.
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af"
_CJK_RUN_RE = re.compile(f"[{_CJK}]+")
_CJK_TOKEN_RE = re.compile(f"[{_CJK}]+$")

#: A Latin/digit word, or a run of CJK. Used by :func:`tokenize` in one pass.
_TOKEN_RE = re.compile(f"[a-z0-9][a-z0-9_+\\-./]*|[{_CJK}]+")

#: Sentence terminators in both scripts.
#:  * Latin ``.!?`` only ends a sentence when whitespace or end-of-text follows, so
#:    "0.4%" does not split. A trailing abbreviation ("Fig. 3") *does* split -- a
#:    general fix needs an abbreviation list -- and that is left as-is because the
#:    resulting four-character fragment is dropped by the answerer's minimum sentence
#:    length before it can reach a user.
#:  * CJK terminators end a sentence unconditionally -- Chinese does not put a
#:    space after 。.
#:  * A blank line is also a boundary, which matters for policy notices that use
#:    headings instead of full stops.
_BOUNDARY_RE = re.compile(r"[.!?]+(?=\s|$)|[。！？；;]|\n\s*\n")

_LATIN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "said",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "which",
        "will",
        "with",
        "would",
        "after",
        "also",
        "but",
        "can",
        "into",
        "more",
        "not",
        "over",
        "than",
        "then",
        "up",
        "we",
        "you",
        "de",
        "la",
        "el",
        "en",
        "y",
    }
)

#: CJK function characters. Kept separate from the Latin list above purely for
#: readability, and merged before use. Deliberately narrow -- only genuinely
#: content-free characters. 年/月/日/大/小/高/低/涨/跌 all carry meaning and stay.
_CJK_STOPWORDS = frozenset(
    "的了和与及在是有为对将已也而或等个这那我你他它们就都很更最请吧呢吗啊之其该此"
    "被把让从向于把并则且矣乎哉曰云"
)

#: The set :func:`tokenize` actually consults.
_STOPWORDS = _LATIN_STOPWORDS | _CJK_STOPWORDS


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip the ends."""
    return _WS_RE.sub(" ", text or "").strip()


def normalize_title(title: str) -> str:
    """Lowercase, de-punctuate and collapse whitespace -- keeping CJK intact.

    Also removes the ``" - Site Name"`` / ``" | Publisher"`` suffix that feeds
    append.

    Note the deliberate absence of an ``encode("ascii", "ignore")`` step. That
    one-liner looks tidy and silently reduces every Chinese policy headline to the
    empty string, which in turn makes every Chinese document share a single
    de-duplication key. NFKC folds width variants (full-width digits, ideographic
    spaces) without deleting anything.
    """
    text = unicodedata.normalize("NFKC", title or "")
    # Unify dash variants (hyphen/en/em/minus) so the suffix rule sees one separator.
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    # Drop a trailing " - publisher" segment, a very common feed artefact.
    text = re.sub(r"\s+[-|]\s+[^-|]{2,40}$", "", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return normalize_whitespace(text)


def normalize_for_match(text: str) -> str:
    """Aggressive normalisation used for content fingerprints.

    Everything that survives is a letter, a digit or a CJK character, so
    formatting-only differences (smart quotes, NBSP, zero-width joiners) do not
    make two copies of the same document look distinct.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = _PUNCT_RE.sub(" ", text.lower())
    return normalize_whitespace(text)


def has_cjk(text: str) -> bool:
    """True when ``text`` contains at least one CJK character."""
    return _CJK_RUN_RE.search(text or "") is not None


def strip_tracking(url: str) -> str:
    """Remove tracking/session query parameters and any fragment from ``url``."""
    parts = urlsplit(url)
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), ""))


# ---------------------------------------------------------------------------
# Identity and de-duplication
# ---------------------------------------------------------------------------


def content_hash(text: str) -> str:
    """Fingerprint of the normalised body text (near-duplicate detection)."""
    return hashlib.sha256(normalize_for_match(text).encode("utf-8")).hexdigest()[:20]


def dedupe_key(title: str, url: str = "") -> str:
    """Tier-2 key: the normalised title, or the URL when the title is empty.

    Catches cross-domain syndication of the same headline, which is the single
    most common duplicate pattern in mining news.
    """
    basis = normalize_title(title) or strip_tracking(url)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def article_id(title: str, url: str) -> str:
    """Tier-1 key: canonical URL plus normalised title.

    Combining both keeps two genuinely different articles that share a headline
    (recurring columns, monthly price reviews) apart, while still collapsing
    URL-only variations of one article.
    """
    basis = f"{strip_tracking(url)}|{normalize_title(title)}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


def document_id(kind: str, url: str, title: str, published: datetime | None = None) -> str:
    """Primary key for a :class:`~pipeline.models.Document`.

    ``kind`` is part of the key because the same URL can legitimately appear as
    both a news article and a policy notice, and the two are different rows.
    """
    stamp = published.date().isoformat() if published else ""
    basis = f"{kind}|{strip_tracking(url)}|{normalize_title(title)}|{stamp}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def shingles(text: str, size: int = 5) -> frozenset[str]:
    """Word-level shingles, used for near-duplicate detection."""
    tokens = tokenize(text, keep_stopwords=True)
    if len(tokens) < size:
        return frozenset(tokens)
    return frozenset(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))


def dedupe_fingerprint(text: str, size: int = 5, keep: int = 1024) -> frozenset[str]:
    """A bounded, deterministic signature of ``text`` for near-duplicate detection.

    Tier 3 of the de-duplication cascade. Tier 1 (URL) and tier 2 (title) both fail
    on the case that actually hurts: a wire story republished under a rewritten
    headline. Comparing 5-word shingles catches it.

    ``keep`` is a *cap*, not a target, and the default is far above the size of a
    realistic chunk. This matters more than it looks: a previous version capped at
    64 shingles, and because the cap keeps the lexicographically smallest ones, a
    one-sentence edit shifted which shingles survived the cut -- two near-identical
    documents scored ~0.7 instead of ~0.95, so the cascade silently missed the
    reprints it exists to catch. The cap is retained only so a pathological
    document (a page that inlines an entire sitemap) cannot allocate unbounded
    memory; ``clean_text`` bounds document length before this is ever reached.
    """
    shingles_ = shingles(text, size)
    if len(shingles_) <= keep:
        return shingles_
    return frozenset(sorted(shingles_)[:keep])


# ---------------------------------------------------------------------------
# Tokenisation and lexical scoring
# ---------------------------------------------------------------------------


def _cjk_ngrams(run: str) -> list[str]:
    """Unigrams **and** bigrams for a CJK run.

    Bigrams alone are the usual choice, but they break single-character queries:
    a user asking about 锂 finds nothing, because the corpus only ever holds 碳酸
    酸锂 锂价. Indexing both means 锂 matches directly, at the cost of some noise
    -- which is what ``_CJK_STOPWORDS`` and the vector half of retrieval are for.
    """
    if len(run) == 1:
        return [run]
    return [*run, *(run[i : i + 2] for i in range(len(run) - 1))]


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Split ``text`` into searchable tokens, CJK-aware.

    Latin and digit words pass through whole. A CJK run is expanded into
    character bigrams: retrieval over Chinese text needs overlapping n-grams,
    because there are no spaces to split on and unigrams are far too ambiguous
    (锂 / 价 / 格 appear in almost every lithium document).
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer((text or "").lower()):
        piece = match.group(0)
        if _CJK_TOKEN_RE.match(piece):
            tokens.extend(_cjk_ngrams(piece))
        else:
            tokens.append(piece)
    if keep_stopwords:
        return tokens
    # A single CJK character is meaningful (锂, 铜, 镍); a single Latin letter is not.
    return [
        token
        for token in tokens
        if token not in _STOPWORDS and (len(token) > 1 or _CJK_TOKEN_RE.match(token))
    ]


def jaccard(left: set[str] | frozenset[str], right: set[str] | frozenset[str]) -> float:
    """Jaccard similarity of two token sets; 0.0 when either side is empty."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# ---------------------------------------------------------------------------
# Bilingual concept lexicon
# ---------------------------------------------------------------------------
#
# The corpus is genuinely bilingual: China's rare-earth and lithium policy is in
# Chinese, Australia's critical-minerals material is in English, and a user asks in
# whichever language they think in. A token-overlap retriever cannot bridge that --
# "澳大利亚关键矿产清单有哪些变化" shares no token with "Australia updates the
# Critical Minerals List" -- and a deterministic feature-hashing embedder cannot
# either, because no amount of hashing knows that 稀土 and "rare earth" mean the same
# thing.
#
# A learned cross-lingual encoder would fix this and is the right answer at scale.
# The answer that fits in this project is a hand-written concept lexicon: ~50 domain
# concepts, each with its Chinese and English surface forms. It is transparent,
# auditable, costs nothing at query time, and it fixes exactly the queries that
# matter here. It is a deliberate, documented limitation -- not a hidden one.

#: Canonical concept -> surface forms. **Forms are written pre-normalised** (lower
#: case, punctuation stripped, dash variants unified), because matching happens
#: against :func:`normalize_for_match` output. Pure-ASCII forms are matched on word
#: boundaries; CJK forms are matched as substrings.
_CONCEPT_LEXICON: dict[str, tuple[str, ...]] = {
    # -- jurisdictions and issuers ---------------------------------------
    "china": ("中国", "国内", "我国", "china", "chinese", "prc"),
    "australia": ("澳大利亚", "澳洲", "西澳", "australia", "australian"),
    "indonesia": ("印尼", "印度尼西亚", "indonesia", "indonesian"),
    "africa": ("刚果", "津巴布韦", "非洲", "congo", "zimbabwe", "africa"),
    "latin_america": ("智利", "阿根廷", "巴西", "chile", "argentina", "brazil"),
    # -- minerals ---------------------------------------------------------
    "rare_earth": ("稀土", "rare earth", "rare earths", "neodymium", "praseodymium", "dysprosium"),
    "lithium": ("锂", "碳酸锂", "氢氧化锂", "锂辉石", "lithium", "lithium carbonate", "spodumene"),
    "copper": ("铜", "copper"),
    "nickel": ("镍", "nickel"),
    "zinc": ("锌", "zinc"),
    "iron_ore": ("铁矿石", "iron ore", "62 fe", "65 fe"),
    "cobalt": ("钴", "cobalt"),
    "graphite": ("石墨", "graphite"),
    "tungsten": ("钨", "tungsten"),
    "manganese": ("锰", "manganese"),
    "gallium": ("镓", "gallium"),
    "germanium": ("锗", "germanium"),
    "molybdenum": ("钼", "molybdenum"),
    "alumina": ("氧化铝", "alumina"),
    "critical_minerals": ("关键矿产", "关键矿物", "critical mineral", "critical minerals"),
    # -- policy instruments ----------------------------------------------
    "quota": ("总量控制", "指标", "配额", "quota", "quotas", "cap"),
    "export_control": ("出口管制", "两用物项", "export control", "export controls", "dual use"),
    "licence": ("许可证", "许可", "审批", "licence", "license", "permit", "approval"),
    "catalogue": ("清单", "目录", "catalogue", "list"),
    "standard": ("标准", "规范", "standard", "specification"),
    "strategy": ("规划", "战略", "strategy", "plan"),
    "consultation": ("征求意见", "公众意见", "consultation", "submission"),
    "subsidy": ("补贴", "扶持", "收储", "储备", "subsidy", "incentive", "financing", "funding"),
    "tax": ("资源税", "税费", "tax", "royalty"),
    # -- market concepts --------------------------------------------------
    "price": ("价格", "报价", "价", "price", "prices", "quotation"),
    "trend": ("走势", "趋势", "行情", "trend", "outlook"),
    "rise": ("上涨", "涨幅", "涨", "rise", "rises", "rose", "up", "gain", "gains"),
    "fall": ("下跌", "跌幅", "跌", "fall", "falls", "fell", "down", "decline", "declines", "loss"),
    "range": ("区间", "波幅", "range", "volatility"),
    "inventory": (
        "库存",
        "存货",
        "去库存",
        "inventory",
        "inventories",
        "stock",
        "stockpile",
        "destocking",
    ),
    "supply": ("供应", "供给", "产量", "产能", "supply", "output", "production", "capacity"),
    "demand": ("需求", "消费", "demand", "consumption"),
    "cost": ("成本", "cost", "costs"),
    "guidance": ("指引", "指导产量", "guidance", "forecast"),
    "project": ("项目", "矿山", "project", "mine", "expansion", "feasibility"),
    "trade": (
        "进口",
        "出口",
        "发运",
        "trade",
        "import",
        "imports",
        "export",
        "exports",
        "shipment",
    ),
    "incentive": ("税收优惠", "生产税收", "tax incentive", "production tax", "credit"),
    "smelter": ("冶炼", "冶炼厂", "加工厂", "smelter", "smelters", "refining", "refinery"),
    "treatment_charge": (
        "加工费",
        "处理费",
        "冶炼费",
        "treatment charge",
        "treatment charges",
        "tc rc",
    ),
    "port": ("港口", "口岸", "港口库存", "port", "ports", "warehouse", "warehouses"),
    "review": ("评估", "审查", "复盘", "review", "assessment"),
    "magnet": ("磁材", "永磁", "磁铁", "magnet", "magnets"),
    "cathode": ("正极材料", "三元材料", "cathode", "cathodes"),
    "closure": (
        "停产",
        "关停",
        "关闭",
        "停产检修",
        "closure",
        "closures",
        "curtailment",
        "curtailments",
    ),
    "low": ("多年低点", "历史低点", "低点", "multi year low", "multi year lows", "record low"),
    "high": ("多年高点", "历史高点", "高点", "multi year high", "multi year highs", "record high"),
    "funding": (
        "资金",
        "融资",
        "支持包",
        "扶持资金",
        "support package",
        "financing",
        "funding",
        "concessional",
    ),
    "utilisation": ("开工率", "产能利用率", "利用率", "utilisation", "utilization"),
}

#: Marker prefix for a concept token. Any character outside ``\w`` works, because
#: real tokens are produced by a ``\w``-based regex and can never start with it.
_CONCEPT_PREFIX = "\u00a7"

#: How much a matched concept counts relative to a matched literal token. 3.0 is a
#: prior, not a tuned value: a concept is a hand-verified equivalence class, so one
#: concept match is worth about as much as a handful of token matches. There is no
#: held-out set here to fit it on, and pretending otherwise would be worse than
#: stating the assumption.
_CONCEPT_WEIGHT = 3.0

#: Pre-compiled matchers, built once. ASCII forms get word boundaries so "list"
#: does not fire on "listing"; CJK forms are matched as substrings because Chinese
#: has no word delimiters to anchor on.
_CONCEPT_MATCHERS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (
        concept,
        tuple(
            re.compile(rf"\b{re.escape(form)}\b") if form.isascii() else re.compile(re.escape(form))
            for form in forms
        ),
    )
    for concept, forms in _CONCEPT_LEXICON.items()
)


def concepts_in(text: str) -> list[str]:
    """Canonical concepts mentioned in ``text``, in lexicon order.

    Lexicon order (not occurrence order) is deliberate: the output is used to build
    token sets and embeddings, and a stable order keeps the embedding of a document
    independent of where in the text a concept happened to appear.
    """
    if not text:
        return []
    normalized = normalize_for_match(text)
    return [
        concept
        for concept, matchers in _CONCEPT_MATCHERS
        if any(matcher.search(normalized) for matcher in matchers)
    ]


def search_tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Tokens **plus** canonical concept markers -- what lexical scoring compares.

    Used instead of :func:`tokenize` everywhere the language of the query and the
    language of the document may differ, which in this corpus is most of the time.
    """
    return [
        *tokenize(text, keep_stopwords=keep_stopwords),
        *(_CONCEPT_PREFIX + concept for concept in concepts_in(text)),
    ]


def search_units(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Word-like units: maximal CJK runs and Latin words, plus concept markers.

    This is the vocabulary lexical *scoring* compares, and it is deliberately not
    the same as :func:`tokenize`. Tokenisation expands a CJK run into unigrams and
    bigrams so that a single-character query can match; that is right for matching
    and wrong for scoring, because it bloats the query. "印度尼西亚" is one word and
    one unit, but nine tokens -- so a candidate document with n-gram overlap on
    three of those nine looks a third relevant, and a document that actually
    contains every one of the query's words can be out-scored by CJK noise.

    Keeping the two vocabularies separate is the fix: :func:`tokenize` decides
    *whether* text matches, :func:`search_units` decides *how much*.
    """
    units: list[str] = []
    for match in _TOKEN_RE.finditer((text or "").lower()):
        piece = match.group(0)
        if _CJK_TOKEN_RE.match(piece):
            if keep_stopwords or piece not in _STOPWORDS:
                units.append(piece)
        elif keep_stopwords or (piece not in _STOPWORDS and len(piece) > 1):
            units.append(piece)
    units.extend(_CONCEPT_PREFIX + concept for concept in concepts_in(text))
    return units


def relevance_score(query: str, text: str) -> float:
    """Score how relevant ``text`` is to ``query`` in ``[0, 1]``.

    Deliberately corpus-free: coverage of query tokens plus a small bonus for the
    whole query appearing verbatim. No IDF, no model, no surprises -- the vector
    store carries the semantic load, this only breaks ties.
    """
    query_tokens = set(search_tokens(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(search_tokens(text))
    covered = query_tokens & text_tokens
    coverage = len(covered) / len(query_tokens)
    bonus = 0.15 if query.strip().lower() in (text or "").lower() else 0.0
    return min(1.0, coverage * 0.85 + bonus)


def lexical_score(
    query: str,
    text: str,
    *,
    full_tf: int = 3,
    df: Mapping[str, int] | None = None,
    n_docs: int = 0,
    doc_units: set[str] | None = None,
) -> float:
    """Lexical match strength in ``[0, 1]``: the exact-term half of hybrid search.

    Scoring happens over :func:`search_units` -- word-like units, not n-grams --
    and has two components:

    * **coverage** -- an IDF-weighted fraction of the query's units that occur in
      the document. This is what makes a rare term ("印度尼西亚", "2027") decisive
      and a ubiquitous one (镍, 政策) nearly free. Without the IDF weighting every
      nickel document in the corpus scores the same for a question about nickel, and
      the terms that actually distinguish the answer -- the country, the year, the
      instrument -- carry no more weight than the mineral name.
    * **density** -- how often the matched units occur, saturating at ``full_tf``.
      A term in the title and the lede is more central than one in a closing
      sentence, but a document repeating a word twenty times is not twenty times
      more relevant.

    Concept matches are counted at :data:`_CONCEPT_WEIGHT` times a literal unit, so
    a Chinese question about 关键矿产 still scores an English document about the
    Critical Minerals List as a strong match.

    ``doc_units`` lets the caller pass a precomputed :func:`search_units` set for the
    document. A search calls this once per candidate, and re-deriving the set each
    time would re-run the concept matchers for every document on every query.
    """
    query_units = list(dict.fromkeys(search_units(query)))
    if not query_units:
        return 0.0
    doc_lower = (text or "").lower()
    if not doc_lower:
        return 0.0
    units = doc_units if doc_units is not None else set(search_units(text))

    def idf(unit: str) -> float:
        if not df or not n_docs:
            return 1.0
        # Smoothed IDF: ~0 for a unit in every document, ~log(n) for a unit in one.
        # The +1 inside the log keeps the weight positive, so a ubiquitous unit still
        # counts for a little.
        return math.log(1.0 + n_docs / (1.0 + df.get(unit, 0)))

    def occurrences(unit: str) -> int:
        # CJK units are matched as substrings, because a document's units are maximal
        # runs: 碳酸锂 is one run, so a query for 锂 would find nothing in a unit set.
        # Latin units match on set membership, which avoids "list" firing on "listing".
        if unit.startswith(_CONCEPT_PREFIX) or unit.isascii():
            return 1 if unit in units else 0
        return doc_lower.count(unit)

    matched = [(unit, occurrences(unit)) for unit in query_units]
    matched = [(unit, count) for unit, count in matched if count]
    if not matched:
        return 0.0

    def weight(unit: str) -> float:
        return _CONCEPT_WEIGHT * idf(unit) if unit.startswith(_CONCEPT_PREFIX) else idf(unit)

    numerator = sum(weight(unit) for unit, _ in matched)
    denominator = sum(weight(unit) for unit in query_units)
    coverage = numerator / denominator if denominator else 0.0
    density = sum(min(1.0, count / full_tf) for _, count in matched) / len(matched)
    return min(1.0, 0.75 * min(1.0, coverage) + 0.25 * density)


# ---------------------------------------------------------------------------
# Sentence splitting and chunking
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split into sentences on both Latin and CJK terminators."""
    return [s for s in (text[start:end].strip() for start, end in _sentence_spans(text)) if s]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of each sentence in ``text``.

    Slicing the original string keeps offsets exact, which is what lets a
    citation point at a real span instead of a re-joined approximation.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _BOUNDARY_RE.finditer(text or ""):
        end = match.end()
        if (text[cursor:end]).strip():
            spans.append((cursor, end))
        cursor = end
    if (text[cursor:]).strip():
        spans.append((cursor, len(text)))
    return spans


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[tuple[str, int, int]]:
    """Split ``text`` into ``(chunk, char_start, char_end)`` triples.

    Sentences are the atomic unit -- a chunk never ends mid-sentence, because a
    citation that reads "the ministry said it would" is worse than useless. A
    single sentence longer than ``size`` is hard-split on a word boundary so
    pathological input still terminates.

    ``overlap`` is honoured by starting the next window at the earliest sentence
    whose start is at least ``overlap`` characters behind the previous window's
    end. That keeps a fact which straddles a boundary retrievable from either
    side, without duplicating whole chunks.
    """
    text = text or ""
    if not text.strip():
        return []

    stripped_start = len(text) - len(text.lstrip())

    def emit(start: int, end: int) -> list[tuple[str, int, int]]:
        if end - start <= size:
            body = text[start:end].strip()
            return [(body, start, end)] if body else []
        return [
            (text[piece_start:piece_end].strip(), piece_start, piece_end)
            for piece_start, piece_end in _hard_split(text, start, end, size)
            if text[piece_start:piece_end].strip()
        ]

    if len(text) <= size:
        body = text[stripped_start:].rstrip()
        return [(body, stripped_start, stripped_start + len(body))] if body else []

    spans = _sentence_spans(text)
    if not spans:
        body = text[stripped_start:].rstrip()
        return [(body, stripped_start, stripped_start + len(body))] if body else []

    chunks: list[tuple[str, int, int]] = []
    index = 0
    total = len(spans)
    while index < total:
        start = spans[index][0]
        end = spans[index][1]
        probe = index + 1
        while probe < total and spans[probe][1] - start <= size:
            end = spans[probe][1]
            probe += 1
        chunks.extend(emit(start, end))

        if probe >= total:
            break
        if overlap > 0:
            # Walk back until the trailing sentences cover `overlap` characters.
            back = probe
            while back > index + 1 and end - spans[back - 1][0] < overlap:
                back -= 1
            index = back
        else:
            index = probe
    return chunks


def _hard_split(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Split an over-long span on whitespace, never exceeding ``size``."""
    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + size, end)
        if stop < end:
            window = text[cursor:stop]
            space = window.rfind(" ")
            if space > size // 2:
                stop = cursor + space
        pieces.append((cursor, stop))
        cursor = stop
    return pieces


# ---------------------------------------------------------------------------
# Dates and misc
# ---------------------------------------------------------------------------


def parse_datetime(value: str | datetime | None) -> datetime | None:
    """Best-effort parse of RFC 822 / ISO 8601 / common RSS and Chinese date strings."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        pass
    for candidate in (raw, raw.replace("Z", "+00:00"), raw.replace("/", "-")):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%Y年%m月%d日",
        "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def slugify(text: str, max_length: int = 60) -> str:
    """Filesystem-safe slug.

    Pure-CJK input has no ASCII characters to keep, so it falls back to a short
    digest rather than returning "untitled" for every Chinese document -- which
    would make two different files collide on one name.
    """
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    if not slug:
        digest = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:10]
        return f"doc-{digest}"[:max_length]
    return slug[:max_length].strip("-")


def truncate(text: str, limit: int, suffix: str = "...") -> str:
    """Truncate to ``limit`` characters on a word boundary when possible."""
    clean = normalize_whitespace(text)
    if len(clean) <= limit:
        return clean
    cut = clean[: max(0, limit - len(suffix))]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + suffix
