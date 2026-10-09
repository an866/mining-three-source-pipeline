"""The evaluation set: 20 questions with a known-correct answer in the corpus.

Why it is written against ``pipeline.synthetic.anchors``
-------------------------------------------------------
An evaluation set is only worth running if its ground truth is *true of the
index*. Generating 700 documents and then hand-writing questions against them is
how a harness ends up measuring nothing. So the corpus is built around a fixed
anchor table, and every question here resolves its expected document through that
table (``anchor_urls``) rather than by pasting a URL. If an anchor moves, this file
fails loudly at import instead of quietly scoring zero.

Question style
--------------
Written the way an analyst actually asks -- Chinese, English, and shorthand --
rather than by paraphrasing the anchor title. That matters: the interesting
failures in this corpus are cross-lingual (a Chinese question about an English
wire story) and exact-term ("62% Fe", "Indonesia 2027"), and a ground truth that
only asks questions in the document's own words would never exercise either.

Metrics
-------
``expected_urls`` drives **recall@k**: a question is a hit if any expected URL is
in the top-k retrieved chunks. ``key_facts`` drives **answer correctness** and, with
:mod:`eval.metrics`, **faithfulness**. Each fact is a group of acceptable
alternatives, because a bilingual corpus will render the same fact as "down" or
"下跌" depending on which sentence the answer quotes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.render import FLAT_THRESHOLD_PCT
from pipeline.synthetic.anchors import PRICE_TARGETS, anchor_urls

__all__ = [
    "PRICE_QUESTIONS",
    "QUESTIONS",
    "REFUSAL_CASES",
    "Question",
    "price_key_facts",
]

#: A fact group: any one of the strings satisfies it.
FactGroup = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Question:
    """One evaluation question and everything needed to score an answer."""

    id: str
    question: str
    #: Any of these URLs appearing in the top-k counts as a retrieval hit.
    expected_urls: tuple[str, ...]
    #: One entry per fact the answer must contain; any alternative in the group passes.
    key_facts: tuple[FactGroup, ...] = ()
    #: What a correct answer looks like, for a human reading the report.
    reference: str = ""
    notes: str = ""
    #: Optional constraint that must hold for the *answer*, not the retrieval.
    expect_any: FactGroup = field(default_factory=tuple)


def price_key_facts(commodity: str) -> tuple[FactGroup, ...]:
    """Facts a correct answer about ``commodity``'s 60-session trend must contain.

    Derived from :data:`pipeline.synthetic.anchors.PRICE_TARGETS` rather than typed
    by hand, so the harness cannot drift from the generator: change the target and
    the expected facts follow.
    """
    target, _ = PRICE_TARGETS[commodity]
    pct = abs(target) * 100.0
    direction = "flat" if pct < FLAT_THRESHOLD_PCT else ("up" if target > 0 else "down")
    word = {"up": "上涨", "down": "下跌", "flat": "基本持平"}[direction]
    return (
        (f"{pct:.2f}",),  # the magnitude, e.g. "18.50"
        (direction, word),  # the direction in either language
    )


def _price_url(commodity: str, sessions: int = 60) -> str:
    return f"https://prices.example/{commodity}/trend-{sessions}d"


# ---------------------------------------------------------------------------
# The 20 questions
# ---------------------------------------------------------------------------

_CN_POLICY: tuple[Question, ...] = (
    Question(
        id="cn-re-quota",
        question="中国稀土集团 2026 年上半年稀土开采总量控制指标是多少？",
        expected_urls=anchor_urls("cn_rare_earth_quota_2026h1"),
        key_facts=(("135000", "135,000", "13.5 万"), ("19150", "19,150"), ("172000", "172,000")),
        reference="岩矿型稀土开采指标 135000 吨，离子型 19150 吨，冶炼分离 172000 吨。",
        notes="exact-term: large Chinese numerals, three separate figures",
    ),
    Question(
        id="cn-export-control",
        question="镓、锗相关物项出口管制从什么时候开始施行？需要办什么手续？",
        expected_urls=anchor_urls("cn_export_control_gallium_germanium"),
        key_facts=(("2026", "9 月 1 日"), ("许可", "licence", "license", "许可")),
        reference="2026 年 9 月 1 日起施行，出口须向商务部申请许可。",
        notes="date + obligation",
    ),
    Question(
        id="cn-lithium-standard",
        question="2026 版碳酸锂产品标准把电池级纯度要求提高到了多少？",
        expected_urls=anchor_urls("cn_lithium_processing_standard"),
        key_facts=(("99.6",), ("99.5",)),
        reference="电池级碳酸锂纯度由 99.5% 提高到 99.6%。",
        notes="numeric comparison: both the old and new threshold",
    ),
    Question(
        id="cn-re-licence",
        question="稀土出口许可证的办理时限调整成了多久？原因是什么？",
        expected_urls=anchor_urls("cn_rare_earth_export_licence_slowdown"),
        key_facts=(("25",), ("15",)),
        reference="由 15 个工作日延长至 25 个工作日，原因是新增最终用户核查。",
        notes="numeric comparison",
    ),
)

_AU_POLICY: tuple[Question, ...] = (
    Question(
        id="au-cml",
        question="澳大利亚关键矿产清单 2026 年更新新增和移除了哪些矿种？",
        expected_urls=anchor_urls("au_disr_critical_minerals_list_update"),
        key_facts=(("gallium", "镓"), ("germanium", "锗"), ("helium", "氦")),
        reference="新增镓、锗、高纯氧化铝，移除氦。",
        notes="cross-lingual: Chinese question, English policy document",
    ),
    Question(
        id="au-lithium-consult",
        question="澳大利亚锂精矿出口许可条件征求意见的核心要求是什么？",
        expected_urls=anchor_urls("au_lithium_export_licence_review"),
        key_facts=(("30",), ("2027",)),
        reference="要求至少 30% 的锂在境内转化，2027 年 7 月 1 日起分阶段实施。",
        notes="cross-lingual; percentage + date",
    ),
    Question(
        id="au-re-package",
        question="澳大利亚为稀土加工提供了多大规模的资金支持？重点支持哪些元素？",
        expected_urls=anchor_urls("au_rare_earth_support_package"),
        key_facts=(("1.2",), ("neodymium", "钕", "praseodymium", "镨")),
        reference="A$1.2bn，重点支持钕、镨、镝、铽。",
        notes="cross-lingual; currency amount",
    ),
    Question(
        id="au-nickel-review",
        question="澳大利亚镍市场评估发现多少比例的镍企业处于现金亏损？",
        expected_urls=anchor_urls("au_nickel_review"),
        key_facts=(("60",),),
        reference="约 60% 的澳大利亚镍企业现金亏损。",
        notes="cross-lingual; percentage",
    ),
)

_NEWS: tuple[Question, ...] = (
    Question(
        id="news-pilbara",
        question="Pilbara Minerals 把锂精矿产量指引上调到了多少？",
        expected_urls=anchor_urls("news_pilbara_guidance"),
        key_facts=(("800,000", "800000"), ("840,000", "840000")),
        reference="上调至 80–84 万吨干公吨。",
        notes="exact-term: large figures with thousands separators",
    ),
    Question(
        id="news-lithium-destock",
        question="锂电去库存进展如何？下游库存降到什么水平？",
        expected_urls=anchor_urls("news_lithium_destocking"),
        key_facts=(("four weeks", "4 weeks", "四周", "4 周"),),
        reference="下游库存由九周降至约四周。",
        notes="cross-lingual; colloquial Chinese vs wire copy",
    ),
    Question(
        id="news-copper-tc",
        question="中国铜冶炼厂为什么同意削减产量？",
        expected_urls=anchor_urls("news_copper_smelter_cuts"),
        key_facts=(("treatment", "加工费", "冶炼费"), ("negative", "负", "转负")),
        reference="现货加工费转负，反映精矿短缺。",
        notes="cross-lingual + exact term 'treatment charge'",
    ),
    Question(
        id="news-lme-copper",
        question="LME 铜库存降到了什么水平？原因是什么？",
        expected_urls=anchor_urls("news_lme_copper_inventory"),
        key_facts=(("six-week", "six week", "六周", "6 周"), ("grid", "电网")),
        reference="降至六周低点，因中国电网投资抵消地产走弱。",
        notes="cross-lingual; hyphenated compound",
    ),
    Question(
        id="news-indonesia-nickel",
        question="印度尼西亚 2027 年镍矿配额调整了多少？",
        expected_urls=anchor_urls("news_indonesia_nickel_quota"),
        key_facts=(("240",), ("275",), ("13", "13%")),
        reference="由 2.75 亿吨下调至 2.4 亿吨湿吨，降幅约 13%。",
        notes="cross-lingual: the question the pre-trim used to delete",
    ),
    Question(
        id="news-iron-ore-port",
        question="中国港口铁矿石库存处于什么水平？",
        expected_urls=anchor_urls("news_iron_ore_port_inventory"),
        key_facts=(("148",), ("four-month", "four month", "四个月", "4 个月")),
        reference="升至 1.48 亿吨的四个月高位。",
        notes="cross-lingual; 中国 as topic, not as a source scope",
    ),
    Question(
        id="news-re-magnet",
        question="稀土磁材出口在 9 月恢复得怎么样？",
        expected_urls=anchor_urls("news_rare_earth_magnet_exports"),
        key_facts=(("18", "18%"), ("september", "9 月")),
        reference="环比回升 18%，仍低于五年均值。",
        notes="cross-lingual; month reference",
    ),
    Question(
        id="news-zinc-tc",
        question="锌加工费为什么跌到多年低点？",
        expected_urls=anchor_urls("news_zinc_tc_collapse"),
        key_facts=(("closure", "关闭", "停产"), ("deficit", "短缺", "赤字")),
        reference="两座中型矿山关闭抽走精矿，冶炼厂减产，后市预期转向短缺。",
        notes="was mis-planned as kind=price via the word 跌",
    ),
)

PRICE_QUESTIONS: tuple[Question, ...] = (
    Question(
        id="price-lithium",
        question="近 60 个交易日碳酸锂价格是涨还是跌？跌幅多少？",
        expected_urls=(_price_url("lithium"),),
        key_facts=price_key_facts("lithium"),
        reference="下跌 18.50%。",
        notes="price series: direction + magnitude",
    ),
    Question(
        id="price-copper",
        question="近 60 个交易日铜价走势如何？",
        expected_urls=(_price_url("copper"),),
        key_facts=price_key_facts("copper"),
        reference="上涨 3.40%。",
        notes="the only benchmark the generator makes rise",
    ),
    Question(
        id="price-nickel",
        question="近 60 个交易日镍价跌幅是多少？",
        expected_urls=(_price_url("nickel"),),
        key_facts=price_key_facts("nickel"),
        reference="下跌 9.20%。",
        notes="price series",
    ),
    Question(
        id="price-iron-ore",
        question="近 60 个交易日铁矿石价格变化大吗？",
        expected_urls=(_price_url("iron_ore"),),
        key_facts=price_key_facts("iron_ore"),
        reference="基本持平，变动不足 1%。",
        notes="exercises the flat band, where 'up or down' is the wrong answer",
    ),
)

QUESTIONS: tuple[Question, ...] = (*_CN_POLICY, *_AU_POLICY, *_NEWS, *PRICE_QUESTIONS)

#: Questions the corpus genuinely cannot answer. The system prompt promises a
#: refusal, so the promise is measured rather than asserted. Recall@k is undefined
#: here and these are scored separately as a refusal rate.
#:
#: Scope of the mechanism: refusal is driven by a *commodity* coverage check (see
#: ``pipeline.retrieval.uncovered_minerals``), so it catches questions about
#: minerals the index does not carry. It does **not** catch a question that names a
#: covered commodity but asks for a fact that is absent -- "智利 2027 年锂矿产量
#: 预测是多少" would still be answered from the nearest lithium evidence. That gap is
#: real and is what the LLM path's refusal clause in ``SYSTEM_PROMPT`` is for; these
#: cases are deliberately limited to the ones the deterministic path can honour.
REFUSAL_CASES: tuple[Question, ...] = (
    Question(
        id="refuse-platinum",
        question="铂金（platinum）最近的价格走势如何？",
        expected_urls=(),
        reference="Corpus covers no platinum series; a correct answer says so.",
        notes="uncovered commodity; measured best hit scores 0.46, above several real answers",
    ),
    Question(
        id="refuse-uranium",
        question="铀矿价格最近走势如何？供应是否紧张？",
        expected_urls=(),
        reference="Corpus carries no uranium series or news.",
        notes="uncovered commodity",
    ),
)
