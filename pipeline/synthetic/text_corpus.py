"""Deterministic synthetic news and policy corpus.

The corpus has two strata:

* **anchors** -- the twelve documents from :mod:`pipeline.synthetic.anchors`,
  emitted verbatim. They carry the facts the evaluation set asks about, so
  recall@5 measures something real rather than measuring the generator.
* **filler** -- template-generated documents that pad each source past the
  brief's ">= 200 items" floor. Fillers are drawn from the same topic space as the
  anchors (lithium, rare earths, copper, nickel, zinc, iron ore, gallium,
  germanium), so retrieval has to *rank* rather than merely *find*.

Everything is seeded from stable integers via ``random.Random(sha256(...))`` --
never the builtin ``hash()``, which is salted per process and would make two runs
over identical inputs produce different corpora, and therefore different eval
scores.
"""

from __future__ import annotations

import hashlib
import random
from datetime import UTC, date, datetime, timedelta
from typing import TypedDict

from pipeline.models import SourceKind
from pipeline.synthetic.anchors import Anchor, anchors_by_kind
from pipeline.synthetic.syndication import syndicate
from pipeline.textutils import normalize_whitespace, slugify

__all__ = [
    "generate_news_records",
    "generate_policy_records",
]


class _Slots(TypedDict):
    """Numeric slots a template or body may reference."""

    pct: int
    pct2: int
    years: int
    bn: float
    days: int
    months: int
    kt: int
    usd: int
    year: int
    year2: int


_MONTH_NAMES: dict[int, str] = {
    1: "January",
    3: "March",
    6: "June",
    7: "July",
    9: "September",
}


# ---------------------------------------------------------------------------
# Topic space -- (key, Chinese noun, English noun)
# ---------------------------------------------------------------------------

_COMMODITIES: tuple[tuple[str, str, str], ...] = (
    ("lithium", "锂", "lithium"),
    ("rare-earth", "稀土", "rare earths"),
    ("copper", "铜", "copper"),
    ("nickel", "镍", "nickel"),
    ("zinc", "锌", "zinc"),
    ("iron-ore", "铁矿石", "iron ore"),
    ("gallium", "镓", "gallium"),
    ("germanium", "锗", "germanium"),
    ("cobalt", "钴", "cobalt"),
    ("graphite", "石墨", "graphite"),
    ("tungsten", "钨", "tungsten"),
    ("manganese", "锰", "manganese"),
)

_REGIONS: tuple[tuple[str, str], ...] = (
    ("Pilbara, Western Australia", "西澳皮尔巴拉"),
    ("the Atacama salar in Chile", "智利阿塔卡马盐湖"),
    ("Salta province in Argentina", "阿根廷萨尔塔省"),
    ("Sulawesi in Indonesia", "印尼苏拉威西"),
    ("Katanga in the Democratic Republic of Congo", "刚果（金）加丹加"),
    ("Jiangxi province", "江西省"),
    ("Sichuan province", "四川省"),
    ("Queensland", "昆士兰州"),
    ("the Northern Territory", "北领地"),
    ("Minas Gerais in Brazil", "巴西米纳斯吉拉斯"),
    ("Quebec", "魁北克"),
    ("the Great Dyke in Zimbabwe", "津巴布韦大岩墙"),
    ("Bayannur in Inner Mongolia", "内蒙古巴彦淖尔"),
    ("Ganzhou in Jiangxi", "江西赣州"),
)

_NEWS_OUTLETS: tuple[str, ...] = (
    "mining.com",
    "mining-technology",
    "sp-global-metals",
    "reuters-metals",
    "fastmarkets-news",
    "smm-news",
    "cnyes-mining",
)

#: ``(angle, English headline template, Chinese headline line)``.
#: Slots available: ``en`` / ``zh`` / ``region`` / ``region_zh`` / ``pct`` /
#: ``pct2`` / ``years`` / ``bn`` / ``days`` / ``months`` / ``kt`` / ``year``.
_ANGLES: tuple[tuple[str, str, str], ...] = (
    (
        "output",
        "{region} {en} output rises {pct}% on recovery gains",
        "{region_zh}{zh}产量因回收率提升而增长 {pct}%",
    ),
    (
        "costs",
        "{en} unit cash costs fall {pct}% quarter on quarter at {region}",
        "{region_zh}{zh}单位现金成本环比下降 {pct}%",
    ),
    (
        "project",
        "Feasibility study confirms a {years}-year mine life at the {region} {en} project",
        "{region_zh}{zh}项目可行性研究确认矿山寿命 {years} 年",
    ),
    (
        "mna",
        "Takeover agreed for a {region} {en} producer, valuing equity at ${bn}bn",
        "{region_zh}{zh}生产商达成收购协议，股权估值 {bn} 十亿美元",
    ),
    (
        "trade",
        "{en} imports into China rise {pct}% as {region} shipments resume",
        "{region_zh}发运恢复，{zh}进口量上升 {pct}%",
    ),
    (
        "logistics",
        "Port congestion at {region} adds {days} days to {en} shipment times",
        "{region_zh}港口拥堵使{zh}发运时间延长 {days} 天",
    ),
    (
        "esg",
        "Community agreement at {region} clears the way for {en} expansion",
        "{region_zh}社区协议为{zh}扩建扫清障碍",
    ),
    (
        "technology",
        "New leaching circuit lifts {en} recovery to {pct2}% at the {region} pilot plant",
        "{region_zh}中试厂新浸出流程将{zh}回收率提升至 {pct2}%",
    ),
    (
        "financing",
        "{en} developer in {region} raises ${bn}bn for construction",
        "{region_zh}{zh}开发商融资 {bn} 十亿美元",
    ),
    (
        "inventory",
        "{en} inventories fall to a {months}-month low, supporting prices",
        "{zh}库存降至 {months} 个月低位，支撑价格",
    ),
    (
        "permit",
        "{region} tightens permitting for {en} operations, adding review steps",
        "{region_zh}收紧{zh}项目审批，新增审查环节",
    ),
    (
        "demand",
        "Battery and grid demand lifts the {year} {en} consumption outlook",
        "电池与电网需求推高 {year} 年{zh}消费预期",
    ),
)

_POLICY_ISSUERS_CN: tuple[str, ...] = (
    "中国稀土集团",
    "工业和信息化部",
    "商务部",
    "自然资源部",
    "国家发展和改革委员会",
    "江西省工业和信息化厅",
    "内蒙古自治区自然资源厅",
)

#: Short forms used inside a document number, e.g. ``工信部〔2026〕第 47 号``.
_ISSUER_CODES_CN: dict[str, str] = {
    "中国稀土集团": "稀土集团",
    "工业和信息化部": "工信部",
    "商务部": "商务部",
    "自然资源部": "自然资源部",
    "国家发展和改革委员会": "发改委",
    "江西省工业和信息化厅": "赣工信",
    "内蒙古自治区自然资源厅": "内蒙自然资源",
}

_POLICY_ISSUERS_AU: tuple[str, ...] = (
    "DISR (Australia)",
    "Geoscience Australia",
    "Department of Foreign Affairs and Trade",
)

#: Minerals a policy document can be *about*, as ``(Chinese, English, commodity key)``.
#: Varying this is what stops the generator from emitting 200 notices that all say
#: "稀土" -- real notices are mineral-specific, and a corpus where one word decides
#: every retrieval is not a corpus, it is a keyword index. The key goes onto the
#: document's ``commodity`` field, which is what makes a commodity-filtered policy
#: query possible at all.
_CN_MINERALS: tuple[tuple[str, str, str], ...] = (
    ("稀土", "rare earth elements", "rare_earth"),
    ("锂", "lithium", "lithium"),
    ("镍", "nickel", "nickel"),
    ("钴", "cobalt", "cobalt"),
    ("石墨", "natural graphite", "graphite"),
    ("钨", "tungsten", "tungsten"),
    ("锰", "manganese", "manganese"),
    ("锗", "germanium", "germanium"),
    ("镓", "gallium", "gallium"),
    ("铜", "copper", "copper"),
    ("锌", "zinc", "zinc"),
    ("铁矿石", "iron ore", "iron_ore"),
    ("钼", "molybdenum", "molybdenum"),
    ("钒", "vanadium", "vanadium"),
    ("钛", "titanium", "titanium"),
    ("萤石", "fluorspar", "fluorspar"),
)

#: Provinces named in provincial-level notices.
_CN_REGIONS: tuple[str, ...] = (
    "江西",
    "内蒙古",
    "四川",
    "云南",
    "湖南",
    "广西",
    "福建",
    "甘肃",
    "青海",
    "新疆",
    "贵州",
    "广东",
)

#: The state-owned or listed producers a notice would bind.
_CN_COMPANIES: tuple[str, ...] = (
    "北方稀土",
    "中国稀土集团",
    "赣锋锂业",
    "天齐锂业",
    "华友钴业",
    "洛阳钼业",
    "盛屯矿业",
    "金钼股份",
    "厦门钨业",
    "宝武资源",
    "五矿资源",
    "紫金矿业",
)

#: ``(doc_type, Chinese title template, English title template)``.
#: Available slots in the Chinese templates: ``topic`` / ``zh_noun`` / ``year`` /
#: ``year2`` / ``ref``. In the English ones: ``en`` / ``year`` / ``year2`` / ``ref``.
#:
#: China and Australia get separate sets. Australia's "Critical Minerals List" is a
#: specific named instrument, and it is already represented as an anchor; generating
#: filler catalogues titled "Critical minerals list update" put a dozen documents
#: with near-verbatim Chinese titles ("关键矿产清单更新公告") into the corpus, where
#: they out-ranked the real instrument for a Chinese query purely on literal
#: overlap. Australian instruments are published as notices, standards, strategies
#: and consultations, so that is what the AU set contains.
_POLICY_DOC_TYPES_CN: tuple[tuple[str, str, str], ...] = (
    ("notice", "关于{topic}有关事项的通知（{ref}）", "Notice on {en} ({ref})"),
    (
        "notice",
        "关于调整{zh_noun}管理政策的公告（{ref}）",
        "Announcement adjusting {en} policy ({ref})",
    ),
    ("standard", "{zh_noun}行业标准发布公告（{ref}）", "{en} industry standard released ({ref})"),
    (
        "strategy",
        "{zh_noun}产业发展规划（{year}—{year2}年）（{ref}）",
        "{en} development plan {year}-{year2} ({ref})",
    ),
    ("catalogue", "关键矿产清单更新公告（{ref}）", "Critical minerals list update ({ref})"),
    (
        "consultation",
        "就{topic}管理办法公开征求意见（{ref}）",
        "Consultation on {en} rules ({ref})",
    ),
)

_POLICY_DOC_TYPES_AU: tuple[tuple[str, str, str], ...] = (
    ("notice", "关于{topic}有关事项的通知（{ref}）", "Notice on {en} ({ref})"),
    (
        "notice",
        "关于调整{zh_noun}管理政策的公告（{ref}）",
        "Announcement adjusting {en} policy ({ref})",
    ),
    ("standard", "{zh_noun}行业标准发布公告（{ref}）", "{en} industry standard released ({ref})"),
    (
        "strategy",
        "{zh_noun}产业发展规划（{year}—{year2}年）（{ref}）",
        "{en} development plan {year}-{year2} ({ref})",
    ),
    (
        "consultation",
        "就{topic}管理办法公开征求意见（{ref}）",
        "Consultation on {en} rules ({ref})",
    ),
)

_POLICY_TOPICS_CN: tuple[str, ...] = (
    "开采总量控制指标",
    "出口许可证核发",
    "冶炼分离产能",
    "资源税征收",
    "绿色矿山建设",
    "战略储备收储",
    "两用物项管制",
    "行业准入条件",
    "产能置换",
    "尾矿库安全",
    "智能制造改造",
    "再生原料循环利用",
    "境外投资备案",
    "产品质量分级",
)

#: ``(subject, commodity key)`` for Australian instruments.
_POLICY_EN_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("lithium concentrates", "lithium"),
    ("rare earth elements", "rare_earth"),
    ("nickel ore", "nickel"),
    ("gallium and germanium", "gallium"),
    ("natural graphite", "graphite"),
    ("tungsten concentrates", "tungsten"),
    ("high-purity alumina", "alumina"),
    ("cobalt intermediates", "cobalt"),
)

_MOODS: tuple[tuple[str, str], ...] = (
    ("cautiously firm", "谨慎偏强"),
    ("well supplied", "供应充足"),
    ("range-bound", "区间震荡"),
    ("tightening", "趋于紧张"),
)


def _pick_clauses(
    rng: random.Random,
    clauses: tuple[str, ...],
    *,
    count: int,
    fields: dict[str, object],
) -> list[str]:
    """A deterministic, rotating window of clauses, interpolated with ``fields``.

    A rotation rather than a random sample: consecutive documents draw consecutive
    windows, so any two documents share at most ``count - 1`` clauses, and the
    interpolated fields make even those differ. A random sample would put the same
    four clauses in two documents often enough to threaten the near-duplicate
    threshold.
    """
    start = rng.randrange(len(clauses))
    picked = [clauses[(start + offset) % len(clauses)] for offset in range(count)]
    return [_safe_format(clause, fields) for clause in picked]


def _safe_format(template: str, fields: dict[str, object]) -> str:
    """Format a template, leaving unknown placeholders untouched.

    Clauses and headline templates share one slot namespace, and a typo in a
    placeholder should degrade to a visible ``{literal}`` in the corpus rather than
    a ``KeyError`` that takes the whole generator down.
    """

    class _Keep(dict[str, object]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_Keep(fields))


#: Chinese ordinals used to number the clauses of a notice.
_CN_NUMERALS: tuple[str, ...] = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")


def _rng(*parts: object) -> random.Random:
    """A deterministic RNG for the given key material."""
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


#: The operational clauses a Chinese policy instrument carries after its headline
#: requirement. Two properties matter:
#:
#: 1. They are long enough that a filler document crosses the 900-character chunk
#:    boundary, so the chunker's offset arithmetic is exercised on real corpus data
#:    rather than only in unit tests.
#: 2. Every clause interpolates document-specific fields. Without that, two notices
#:    sharing a clause would share ~90 characters verbatim, and enough shared clauses
#:    would push a pair of genuinely different documents over the near-duplicate
#:    threshold. The corpus probe in DATA_NOTES.md pins down the resulting margin.
_CN_CLAUSES: tuple[str, ...] = (
    "申报材料包括{mineral}项目的采矿许可证、安全生产许可证、近三年产量与销售报表、"
    "环境影响评价批复文件、能耗核查报告，以及{region}省级主管部门要求的资源综合利用方案。",
    "{region}省级主管部门应当在收到完整申报材料后 {days} 个工作日内完成初审，"
    "并将初审意见与{company}等企业的申报材料一并报送，逾期未报送的视为放弃本批次{mineral}指标申请。",
    "对{mineral}产能低于 {kt} 千吨、不符合产业政策或能耗标准的项目，一律不予受理；"
    "已受理的，予以撤回并书面告知{company}。",
    "{company}应当对申报材料的真实性负责。经核查存在弄虚作假情形的，"
    "取消其当年度及下一年度{mineral}申报资格，并按上限 {pct} 万元处罚。",
    "涉及{mineral}总量控制指标的，应当在指标管理信息系统中登记，"
    "未登记的指标不得用于生产、销售或出口，{region}省内企业一并适用。",
    "本文件涉及{mineral}的条款自 {year} 年起施行。此前有关规定与本文件不一致的，"
    "以本文件为准；{company}已取得的既有许可不受影响。",
    "{region}省内{mineral}新建项目应当同步落实能耗与环保措施，"
    "投产后 {days} 个工作日内完成产能与产量备案。",
    "对{mineral}进口原料转为国内加工的，{company}可按规定申报加工贸易手续，"
    "并由{region}主管部门出具核查意见。",
    "{mineral}行业年度报告应当在每年第一季度报送，内容包括产量、库存、"
    "出口量及 {pct}% 产能利用率区间。",
    "对未按规定报送{mineral}数据的企业，由{issuer}责令限期改正；逾期不改正的，纳入行业信用记录。",
)

_EN_CLAUSES: tuple[str, ...] = (
    "Applications must be accompanied by the tenement register extract, the most "
    "recent three years of production and sales returns for {en}, the environmental "
    "approval, and a statement of energy use per unit of output.",
    "The administering agency will assess a complete application within {days} working "
    "days and may request further information once. A request that is not answered "
    "within the stated period is treated as withdrawn.",
    "Projects producing less than {kt}kt per annum, or that do not meet the applicable "
    "industry or energy-efficiency standard, will not be approved; an approval already "
    "granted may be revoked with written reasons.",
    "Applicants are responsible for the accuracy of the material they lodge. Material "
    "misstatement results in loss of eligibility for {en} for the current and following "
    "year, and a penalty of up to {usd} thousand dollars.",
    "Approvals covering {en} are recorded on the public register. Production, sale or "
    "export without a recorded approval is not permitted.",
    "This instrument takes effect for {en} from {year}. Where it is inconsistent with an "
    "earlier instrument, this instrument prevails, and approvals already granted are "
    "unaffected.",
    "Roughly {pct}% of the current {en} project pipeline is expected to require a fresh "
    "assessment, and applicants should plan for {days} additional working days.",
    "Holders must report annual {en} production, inventory and export volumes within the "
    "first quarter of each year, together with a capacity utilisation figure.",
    "A holder that fails to report {en} data as required will be directed to remedy the "
    "omission, and continued non-compliance is recorded on the public register.",
    "Nothing in this instrument restricts the export of {en} to any particular country; "
    "the requirements apply equally to all applicants.",
)


def _number_slots(rng: random.Random, year: int) -> _Slots:
    """The numeric slots every headline and body may reference."""
    return _Slots(
        pct=rng.choice([3, 4, 5, 6, 7, 8, 11, 12, 14, 18]),
        pct2=rng.choice([61, 64, 68, 72, 76, 81, 84]),
        years=rng.choice([8, 10, 12, 15, 20]),
        bn=round(rng.uniform(0.2, 3.4), 1),
        days=rng.choice([3, 5, 7, 10, 14, 21]),
        months=rng.choice([3, 4, 5, 6]),
        kt=rng.choice([20, 35, 50, 80, 120, 200]),
        usd=rng.choice([310, 420, 505, 640, 720, 880]),
        year=year,
        year2=year + 5,
    )


def _spread_date(index: int, total: int, *, days: int, end: date) -> datetime:
    """Spread ``total`` items across the window, newest first.

    Deterministic: item ``i`` lands at a fixed offset with a stable hour, so the
    corpus is not a flat list of midnight timestamps and the "近 7 天" filter has
    something to actually filter.
    """
    offset = 0 if total <= 1 else int(index * (days - 1) / (total - 1))
    seed = hashlib.sha256(f"time:{index}".encode()).hexdigest()
    hour = 6 + int(seed[:2], 16) % 13
    minute = int(seed[2:4], 16) % 60
    return datetime.combine(end - timedelta(days=offset), datetime.min.time(), UTC).replace(
        hour=hour, minute=minute
    )


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


def _news_filler_body(
    *,
    commodity_en: str,
    commodity_zh: str,
    region_en: str,
    region_zh: str,
    angle: str,
    slots: _Slots,
    mood: tuple[str, str],
) -> str:
    """A bilingual body with concrete, quotable numbers."""
    pct, pct2 = slots["pct"], slots["pct2"]

    if angle == "output":
        extra_en = (
            f"Recoveries improved to {pct2}%, and the company guided to a further {pct}% uplift."
        )
        extra_zh = f"回收率提升至 {pct2}%，公司指引再提升 {pct}%。"
    elif angle == "costs":
        extra_en = (
            f"Unit cash costs of ${slots['usd']}/t benefited from lower diesel and reagent "
            f"prices, and a further {pct}% reduction is expected."
        )
        extra_zh = f"单位现金成本 ${slots['usd']}/t，预计再降 {pct}%。"
    elif angle == "inventory":
        extra_en = (
            f"Visible inventories fell to {slots['months']} months of consumption, a "
            f"{pct}% month-on-month drawdown."
        )
        extra_zh = f"可见库存降至 {slots['months']} 个月用量，环比下降 {pct}%。"
    elif angle == "permit":
        extra_en = (
            f"Applicants now face {slots['days']} additional working days of review, "
            f"affecting roughly {pct}% of the pipeline."
        )
        extra_zh = f"申请人需额外 {slots['days']} 个工作日，影响约 {pct}% 的项目。"
    elif angle == "demand":
        extra_en = (
            f"Consumption is forecast to grow {pct}% in {slots['year']}, led by battery and "
            f"grid investment, while supply grows {pct2}%."
        )
        extra_zh = f"{slots['year']} 年消费预计增长 {pct}%，供应增长 {pct2}%。"
    else:
        extra_en = (
            f"The development adds roughly {slots['kt']}kt of {commodity_en} capacity over "
            f"a {slots['days']}-month construction period."
        )
        extra_zh = f"预计新增 {slots['kt']} 千吨{commodity_zh}产能，建设期 {slots['days']} 个月。"

    return normalize_whitespace(
        f"Operations at {region_en} reported a {pct}% change in {commodity_en} metrics over "
        f"the period, according to the company's latest disclosure. {extra_en} Analysts "
        f"covering the region said the figures were broadly in line with consensus but noted "
        f"that the {commodity_en} market remains {mood[0]}.\n"
        f"{region_zh}{commodity_zh}业务在报告期内变动 {pct}%，公司最新披露显示。{extra_zh}"
        f"覆盖该地区的分析师认为数据基本符合预期，但指出{commodity_zh}市场情绪仍{mood[1]}。"
    )


def generate_news_records(
    *,
    days: int,
    target_per_source: int,
    end: date | None = None,
    syndicated: bool = True,
) -> list[dict[str, object]]:
    """News documents: the anchors first, then filler, then syndicated reprints."""
    last = end or datetime.now(UTC).date()
    anchors = anchors_by_kind(SourceKind.NEWS)
    records: list[dict[str, object]] = [_anchor_record(anchor, last) for anchor in anchors]

    filler = max(0, target_per_source - len(anchors))
    generated: list[dict[str, object]] = []
    for index in range(filler):
        rng = _rng("news", index)
        commodity_key, commodity_zh, commodity_en = _COMMODITIES[index % len(_COMMODITIES)]
        region_en, region_zh = _REGIONS[rng.randrange(len(_REGIONS))]
        angle, title_en_tpl, title_zh_tpl = _ANGLES[rng.randrange(len(_ANGLES))]
        outlet = _NEWS_OUTLETS[rng.randrange(len(_NEWS_OUTLETS))]
        mood = _MOODS[rng.randrange(len(_MOODS))]
        slots = _number_slots(rng, last.year)

        slots_all: dict[str, object] = {
            **slots,
            "en": commodity_en,
            "zh": commodity_zh,
            "region": region_en,
            "region_zh": region_zh,
        }
        title = title_en_tpl.format(**slots_all)
        title_zh = title_zh_tpl.format(**slots_all)
        body = _news_filler_body(
            commodity_en=commodity_en,
            commodity_zh=commodity_zh,
            region_en=region_en,
            region_zh=region_zh,
            angle=angle,
            slots=slots,
            mood=mood,
        )
        url = (
            f"https://www.{slugify(outlet)}.example/{angle}/"
            f"{commodity_key}-{slugify(region_en)}-{index}"
        )
        generated.append(
            {
                "kind": SourceKind.NEWS,
                "title": f"{title} - {outlet}",
                "url": url,
                "text": normalize_whitespace(f"{title}. {body}\n{title_zh}。"),
                "published_at": _spread_date(index, filler, days=days, end=last),
                "source": outlet,
                "language": "en",
                "commodity": commodity_key,
                "tags": ["news", commodity_key, angle],
            }
        )
    records.extend(generated)
    if syndicated:
        records.extend(syndicate(generated))
    return records


def _anchor_record(anchor: Anchor, end: date) -> dict[str, object]:
    """Turn an anchor into a corpus record, padded to a realistic length.

    Anchors keep their URL verbatim -- the evaluation harness resolves expected
    URLs to documents, so any rewriting here would silently break recall@5.
    """
    padding = ""
    if anchor.filler_topics:
        padding = (
            " Related considerations include "
            + ", ".join(anchor.filler_topics)
            + ". Further detail is expected in the next reporting period."
        )
    published = datetime.combine(
        end - timedelta(days=anchor.age_days), datetime.min.time(), UTC
    ).replace(hour=9, minute=30)
    return {
        "kind": anchor.kind,
        "title": anchor.title,
        "url": anchor.url,
        "text": normalize_whitespace(f"{anchor.title}. {anchor.body}{padding}"),
        "published_at": published,
        "source": _anchor_source(anchor),
        "language": "zh" if _is_chinese(anchor.title) else "en",
        "jurisdiction": anchor.jurisdiction,
        "issuer": anchor.issuer,
        "doc_type": anchor.doc_type,
        "commodity": anchor.commodities[0] if anchor.commodities else "",
        "tags": [anchor.kind.value, *anchor.commodities],
    }


def _anchor_source(anchor: Anchor) -> str:
    """The logical source name an anchor belongs to."""
    if anchor.kind is SourceKind.NEWS:
        host = anchor.url.split("//", 1)[1].split("/", 1)[0]
        return host.removeprefix("www.")
    if anchor.jurisdiction == "CN":
        return "creg.com.cn" if "creg" in anchor.url else "miit.gov.cn"
    return "industry.gov.au"


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def generate_policy_records(
    *,
    days: int,
    target_per_source: int,
    end: date | None = None,
    syndicated: bool = True,
) -> list[dict[str, object]]:
    """Policy documents: the anchors first, then deterministic filler, then reprints.

    The filler is deliberately *specific* -- a mineral, a province, a named
    producer, a document number -- because a policy corpus where every notice is
    "about rare earths" collapses under any reasonable de-duplication threshold and
    makes retrieval a keyword lookup. Specificity here is what makes both the
    cascade and recall@5 mean something.
    """
    last = end or datetime.now(UTC).date()
    anchors = anchors_by_kind(SourceKind.POLICY)
    records: list[dict[str, object]] = [_anchor_record(anchor, last) for anchor in anchors]

    filler = max(0, target_per_source - len(anchors))
    generated: list[dict[str, object]] = []
    for index in range(filler):
        rng = _rng("policy", index)
        # Two of every three items are Chinese, so both jurisdictions are well
        # represented and the bilingual tokenisation path is genuinely exercised.
        is_cn = index % 3 != 2
        templates = _POLICY_DOC_TYPES_CN if is_cn else _POLICY_DOC_TYPES_AU
        doc_type, title_zh_tpl, title_en_tpl = templates[rng.randrange(len(templates))]
        slots = _number_slots(rng, last.year)
        effective_month = rng.choice([1, 3, 6, 7, 9])
        # A document number makes the item identifiable the way a real 发文字号 does,
        # and guarantees the generated titles are distinct.
        ref = f"{slots['year']}-{100 + index}"

        if is_cn:
            issuer = _POLICY_ISSUERS_CN[rng.randrange(len(_POLICY_ISSUERS_CN))]
            mineral_zh, mineral_en, commodity = _CN_MINERALS[index % len(_CN_MINERALS)]
            topic = _POLICY_TOPICS_CN[rng.randrange(len(_POLICY_TOPICS_CN))]
            region = _CN_REGIONS[rng.randrange(len(_CN_REGIONS))]
            company = _CN_COMPANIES[rng.randrange(len(_CN_COMPANIES))]
            fields: dict[str, object] = {
                **slots,
                "ref": ref,
                "mineral": mineral_zh,
                "zh_noun": mineral_zh,
                "en": mineral_en,
                "region": region,
                "company": company,
                "issuer": issuer,
                "topic": topic,
            }
            title = _safe_format(title_zh_tpl, fields)
            body = normalize_whitespace(
                "".join(
                    [
                        f"{issuer}发布《{title}》，自 {slots['year']} 年 "
                        f"{effective_month} 月 1 日起施行。",
                        f"文件明确：{region}省内{company}等相关企业应当在 "
                        f"{slots['days']} 个工作日内完成申报；",
                        f"涉及{mineral_zh}产能 {slots['kt']} 千吨以上的项目需另行备案；",
                        f"对未按规定执行的企业，处以上限 {slots['pct']} 万元罚款"
                        f"并纳入行业信用记录。",
                        "文件同时说明，本文件施行前已取得许可的项目不受影响。",
                    ]
                )
            )
            clauses = _pick_clauses(rng, _CN_CLAUSES, count=4, fields=fields)
            detail = "".join(
                f"{_CN_NUMERALS[offset]}、{clause}" for offset, clause in enumerate(clauses)
            )
            jurisdiction, language = "CN", "zh"
            source = "creg.com.cn" if issuer == "中国稀土集团" else "miit.gov.cn"
            text = f"{title}。{body}{detail}"
        else:
            issuer = _POLICY_ISSUERS_AU[rng.randrange(len(_POLICY_ISSUERS_AU))]
            subject, commodity = _POLICY_EN_SUBJECTS[rng.randrange(len(_POLICY_EN_SUBJECTS))]
            fields = {
                **slots,
                "ref": ref,
                "en": subject,
                "js": subject,
                "zh_noun": subject,
            }
            title = _safe_format(title_en_tpl, fields)
            body = normalize_whitespace(
                f"{issuer} has released this {doc_type} covering {subject}. It takes effect on "
                f"1 {_MONTH_NAMES[effective_month]} {slots['year']} and applies to projects "
                f"producing more than {slots['kt']}kt per annum. Applicants must lodge supporting "
                f"material within {slots['days']} working days. Roughly {slots['pct']}% of the "
                f"current project pipeline is expected to be affected; existing approvals are "
                f"unaffected and will be grandfathered."
            )
            clauses = _pick_clauses(rng, _EN_CLAUSES, count=3, fields=fields)
            detail = " ".join(clauses)
            jurisdiction, language = "AU", "en"
            source = "industry.gov.au"
            text = f"{title}. {body} {detail}"

        generated.append(
            {
                "kind": SourceKind.POLICY,
                "title": title,
                "url": f"https://www.{slugify(source)}.example/{doc_type}/{jurisdiction.lower()}-{index}",
                "text": normalize_whitespace(text),
                "published_at": _spread_date(index, filler, days=days, end=last),
                "source": source,
                "language": language,
                "jurisdiction": jurisdiction,
                "issuer": issuer,
                "doc_type": doc_type,
                "commodity": commodity,
                "tags": ["policy", jurisdiction.lower(), doc_type, commodity],
            }
        )
    records.extend(generated)
    if syndicated:
        records.extend(syndicate(generated))
    return records
