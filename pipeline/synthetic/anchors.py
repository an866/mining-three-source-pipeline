"""The planted facts that the synthetic corpus and the evaluation set share.

Why this module exists
----------------------
An evaluation set is only meaningful if the ground truth is *true of the corpus*.
Generating 600 documents and then hand-writing 20 questions against them is how
you end up with a harness that measures nothing.

So the corpus is built around this table. Each :class:`Anchor` is a specific,
quotable fact with a fixed URL, and the eval questions in ``eval/ground_truth``
reference those URLs. ``eval.metrics`` resolves an expected URL to the chunks it
produced, which makes recall@5 a real measurement rather than a vibe.

The anchors are modelled on real instruments and real market structure --
Australia's Critical Minerals List, China's rare-earth quotas, LME/SHFE contracts
-- but the numbers are synthetic. DATA_NOTES.md says so plainly; a reader must
never mistake this corpus for collected data.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.models import SourceKind

__all__ = ["ANCHORS", "Anchor", "anchor_urls", "anchors_by_kind"]


@dataclass(frozen=True, slots=True)
class Anchor:
    """One planted fact, and the document that carries it."""

    key: str
    kind: SourceKind
    url: str
    title: str
    body: str
    #: Days before the run date. Keeps anchors inside the 30-day window.
    age_days: int = 3
    jurisdiction: str = ""
    issuer: str = ""
    doc_type: str = ""
    commodities: tuple[str, ...] = ()
    #: Extra sentences appended by the generator to pad the document body.
    filler_topics: tuple[str, ...] = ()

    @property
    def slug(self) -> str:
        return self.key.replace("_", "-")


# ---------------------------------------------------------------------------
# Policy anchors -- China
# ---------------------------------------------------------------------------

_POLICY_CN: tuple[Anchor, ...] = (
    Anchor(
        key="cn_rare_earth_quota_2026h1",
        kind=SourceKind.POLICY,
        url="https://www.creg.com.cn/notice/2026-h1-rare-earth-quota",
        title="中国稀土集团公布 2026 年上半年稀土开采、冶炼分离总量控制指标",
        body=(
            "中国稀土集团公布 2026 年上半年稀土开采、冶炼分离总量控制指标。"
            "上半年岩矿型稀土（以轻稀土为主）开采指标为 135000 吨，"
            "离子型稀土（以中重稀土为主）开采指标为 19150 吨；"
            "冶炼分离产品指标合计 172000 吨，与 2025 年下半年持平。"
            "集团表示将严格执行总量控制制度，加大对非法开采和违规冶炼的查处力度。"
        ),
        age_days=2,
        jurisdiction="CN",
        issuer="中国稀土集团",
        doc_type="notice",
        commodities=("rare_earth", "lithium"),
        filler_topics=("quota", "compliance"),
    ),
    Anchor(
        key="cn_export_control_gallium_germanium",
        kind=SourceKind.POLICY,
        url="https://www.miit.gov.cn/notice/export-control-gallium-germanium-2026",
        title="关于对镓、锗相关物项实施出口管制的公告（2026 年修订）",
        body=(
            "商务部、海关总署会同有关部门，对镓、锗相关物项实施出口管制，"
            "自 2026 年 9 月 1 日起施行。出口经营者应当依照规定向商务部申请许可，"
            "说明最终用途与最终用户。未经许可，不得出口。"
            "涉及物项包括金属镓、氮化镓、氧化锗等八类。"
        ),
        age_days=5,
        jurisdiction="CN",
        issuer="商务部 / 海关总署",
        doc_type="notice",
        commodities=("gallium", "germanium"),
        filler_topics=("licence", "end-use"),
    ),
    Anchor(
        key="cn_lithium_processing_standard",
        kind=SourceKind.POLICY,
        url="https://www.miit.gov.cn/standard/lithium-carbonate-grade-2026",
        title="碳酸锂、氢氧化锂产品标准（2026 版）发布",
        body=(
            "工业和信息化部批准发布碳酸锂、氢氧化锂产品标准 2026 版，"
            "自 2027 年 1 月 1 日起实施。新版标准将电池级碳酸锂的纯度要求"
            "由 99.5% 提高至 99.6%，并新增对磁性异物含量的限值。"
            "标准实施后，不符合要求的产品不得用于动力电池生产。"
        ),
        age_days=9,
        jurisdiction="CN",
        issuer="工业和信息化部",
        doc_type="standard",
        commodities=("lithium",),
        filler_topics=("standard", "battery"),
    ),
    Anchor(
        key="cn_rare_earth_export_licence_slowdown",
        kind=SourceKind.POLICY,
        url="https://www.creg.com.cn/notice/export-licence-processing-time",
        title="稀土出口许可证办理时限调整说明",
        body=(
            "中国稀土集团发布说明，稀土出口许可证的平均办理时限由 15 个工作日"
            "延长至 25 个工作日，主要原因是新增了最终用户核查环节。"
            "说明称该调整适用于全部稀土氧化物及金属产品，不针对特定国家和地区。"
        ),
        age_days=12,
        jurisdiction="CN",
        issuer="中国稀土集团",
        doc_type="notice",
        commodities=("rare_earth",),
        filler_topics=("licence",),
    ),
)

# ---------------------------------------------------------------------------
# Policy anchors -- Australia
# ---------------------------------------------------------------------------

_POLICY_AU: tuple[Anchor, ...] = (
    Anchor(
        key="au_disr_critical_minerals_list_update",
        kind=SourceKind.POLICY,
        url="https://www.industry.gov.au/publications/critical-minerals-list-2026-update",
        title="Australia updates the Critical Minerals List (2026)",
        body=(
            "The Department of Industry, Science and Resources has updated the "
            "Critical Minerals List, adding gallium, germanium and high-purity "
            "alumina, and removing helium. Lithium, rare earth elements, nickel, "
            "cobalt and tungsten remain on the list. Projects producing a listed "
            "mineral become eligible for the A$2bn Critical Minerals Facility and "
            "the 10-year Critical Minerals Production Tax Incentive."
        ),
        age_days=1,
        jurisdiction="AU",
        issuer="DISR (Australia)",
        doc_type="strategy",
        commodities=("gallium", "germanium", "lithium", "rare_earth", "nickel"),
        filler_topics=("facility", "eligibility"),
    ),
    Anchor(
        key="au_lithium_export_licence_review",
        kind=SourceKind.POLICY,
        url="https://www.industry.gov.au/consultation/lithium-export-licence-review",
        title="Consultation opens on lithium concentrate export licence conditions",
        body=(
            "DISR has opened consultation on conditions attached to lithium "
            "concentrate export licences. The proposed change would require "
            "exporters to demonstrate that at least 30% of contained lithium is "
            "converted domestically before shipping, phased in from 1 July 2027. "
            "Submissions close in 60 days. The Department states that existing "
            "long-term offtake contracts will be grandfathered for five years."
        ),
        age_days=4,
        jurisdiction="AU",
        issuer="DISR (Australia)",
        doc_type="consultation",
        commodities=("lithium",),
        filler_topics=("offtake", "grandfathering"),
    ),
    Anchor(
        key="au_rare_earth_support_package",
        kind=SourceKind.POLICY,
        url="https://www.industry.gov.au/news/rare-earth-support-package",
        title="A$1.2bn support package for rare earth processing announced",
        body=(
            "The Australian Government will provide A$1.2bn in concessional "
            "financing for rare earth separation and magnet manufacturing, "
            "targeting neodymium, praseodymium, dysprosium and terbium. The "
            "package supports two planned separation facilities and is intended to "
            "reduce reliance on a single supplier for heavy rare earths."
        ),
        age_days=14,
        jurisdiction="AU",
        issuer="DISR (Australia)",
        doc_type="notice",
        commodities=("rare_earth",),
        filler_topics=("financing",),
    ),
    Anchor(
        key="au_nickel_review",
        kind=SourceKind.POLICY,
        url="https://www.industry.gov.au/review/nickel-market-review",
        title="Nickel market review finds 60% of Australian operations loss-making",
        body=(
            "A departmental review of the nickel market found that roughly 60% of "
            "Australian nickel operations were cash-negative in the preceding "
            "year, attributing the pressure to rapid Indonesian supply growth. "
            "The review recommends a production tax credit and does not recommend "
            "import restrictions."
        ),
        age_days=20,
        jurisdiction="AU",
        issuer="DISR (Australia)",
        doc_type="strategy",
        commodities=("nickel",),
        filler_topics=("tax-credit",),
    ),
)

# ---------------------------------------------------------------------------
# News anchors
# ---------------------------------------------------------------------------

_NEWS: tuple[Anchor, ...] = (
    Anchor(
        key="news_pilbara_guidance",
        kind=SourceKind.NEWS,
        url="https://www.mining.com/pilbara-minerals-raises-spodumene-guidance-2026/",
        title="Pilbara Minerals raises spodumene guidance after Pilgan debottlenecking",
        body=(
            "Pilbara Minerals raised full-year spodumene concentrate guidance to "
            "between 800,000 and 840,000 dry metric tonnes, up from 760,000 to "
            "800,000, after throughput work at the Pilgan plant lifted recoveries "
            "above design rates. The company said unit cash costs fell 9% quarter "
            "on quarter on lower diesel prices and reagent optimisation."
        ),
        age_days=1,
        commodities=("lithium",),
        filler_topics=("guidance", "costs"),
    ),
    Anchor(
        key="news_lithium_destocking",
        kind=SourceKind.NEWS,
        url="https://www.mining.com/lithium-destocking-nears-end-as-cathode-rates-rise/",
        title="Lithium destocking nears its end as cathode utilisation recovers",
        body=(
            "Cathode makers reported utilisation rates recovering to the mid-70s "
            "percent, and downstream inventories have fallen to roughly four weeks "
            "of consumption from nine weeks at the start of the year. Several "
            "conversion plants in Jiangxi restarted idled capacity, though traders "
            "cautioned that the recovery is concentrated in the top five producers."
        ),
        age_days=6,
        commodities=("lithium",),
        filler_topics=("inventory",),
    ),
    Anchor(
        key="news_copper_smelter_cuts",
        kind=SourceKind.NEWS,
        url="https://www.mining.com/chinese-smelters-agree-treatment-charge-cuts/",
        title="Chinese copper smelters agree to deeper treatment charge cuts",
        body=(
            "A group of major Chinese copper smelters agreed to cut output during "
            "the next quarter after spot treatment and refining charges turned "
            "negative, reflecting a concentrate shortage. Imported copper "
            "concentrate arrivals fell 6% month on month, and several plants "
            "brought forward maintenance."
        ),
        age_days=3,
        commodities=("copper",),
        filler_topics=("concentrate",),
    ),
    Anchor(
        key="news_lme_copper_inventory",
        kind=SourceKind.NEWS,
        url="https://www.mining.com/lme-copper-inventories-fall-to-six-week-low/",
        title="LME copper inventories fall to a six-week low as grid demand holds",
        body=(
            "Exchange-registered copper inventories fell to their lowest level in "
            "six weeks, led by drawdowns at Asian warehouses. Analysts attributed "
            "the drawdown to grid investment in China offsetting continued weakness "
            "in residential construction, and noted that cancelled warrants rose to "
            "22% of the total."
        ),
        age_days=2,
        commodities=("copper",),
        filler_topics=("inventory", "grid"),
    ),
    Anchor(
        key="news_indonesia_nickel_quota",
        kind=SourceKind.NEWS,
        url="https://www.mining.com/indonesia-trims-nickel-ore-quota-for-2027/",
        title="Indonesia trims nickel ore quota for 2027",
        body=(
            "Indonesia's energy and mineral resources ministry cut the 2027 nickel "
            "ore production quota to 240 million wet metric tonnes from 275 million, "
            "a cut of roughly 13%. The ministry cited reserve depletion and "
            "environmental compliance. Indonesian nickel supply has already pushed "
            "global prices to multi-year lows."
        ),
        age_days=7,
        commodities=("nickel",),
        filler_topics=("quota", "supply"),
    ),
    Anchor(
        key="news_iron_ore_port_inventory",
        kind=SourceKind.NEWS,
        url="https://www.mining.com/port-hedland-iron-ore-inventories-build/",
        title="Iron ore inventories at Chinese ports build to a four-month high",
        body=(
            "Iron ore inventories at 45 Chinese ports rose to a four-month high of "
            "148 million tonnes, up 5% month on month, as arrivals outpaced "
            "steadily weakening steel mill demand. The Mysteel 62% Fe CFR Qingdao "
            "index drifted sideways, holding within a two-percent range for the "
            "month."
        ),
        age_days=5,
        commodities=("iron_ore",),
        filler_topics=("inventory", "steel"),
    ),
    Anchor(
        key="news_rare_earth_magnet_exports",
        kind=SourceKind.NEWS,
        url="https://www.mining.com/rare-earth-magnet-exports-recover-in-september/",
        title="Rare earth magnet exports recover in September after licence delays",
        body=(
            "Exports of rare earth permanent magnets recovered 18% month on month "
            "in September after a backlog of licence applications was cleared, "
            "although volumes remain below the five-year average. Neodymium-"
            "praseodymium oxide prices firmed on restocking by magnet makers."
        ),
        age_days=8,
        commodities=("rare_earth",),
        filler_topics=("exports",),
    ),
    Anchor(
        key="news_zinc_tc_collapse",
        kind=SourceKind.NEWS,
        url="https://www.mining.com/zinc-treatment-charges-hit-multi-year-lows/",
        title="Zinc treatment charges hit multi-year lows on mine closures",
        body=(
            "Zinc spot treatment charges fell to multi-year lows after the closure "
            "of two mid-sized mines removed concentrate from the market. Smelters "
            "in Europe and South Korea have announced curtailments, and analysts "
            "expect the refined market to move into deficit next year."
        ),
        age_days=11,
        commodities=("zinc",),
        filler_topics=("deficit",),
    ),
)

ANCHORS: tuple[Anchor, ...] = (*_POLICY_CN, *_POLICY_AU, *_NEWS)


def anchors_by_kind(kind: SourceKind) -> tuple[Anchor, ...]:
    """Every anchor belonging to one source family."""
    return tuple(anchor for anchor in ANCHORS if anchor.kind is kind)


def anchor_urls(*keys: str) -> tuple[str, ...]:
    """Resolve anchor keys to URLs, so tests never hard-code a URL string."""
    wanted = set(keys)
    resolved = {anchor.key: anchor.url for anchor in ANCHORS if anchor.key in wanted}
    missing = wanted - resolved.keys()
    if missing:
        raise KeyError(f"unknown anchor key(s): {sorted(missing)}")
    return tuple(resolved[key] for key in keys)


#: Commodities whose price direction the generator guarantees, so that
#: direction-based eval questions have a stable answer on any run date.
#: ``(target 30-session change, tolerance band)``.
PRICE_TARGETS: dict[str, tuple[float, float]] = {
    "lithium": (-0.185, 0.015),
    "nickel": (-0.092, 0.010),
    "zinc": (-0.045, 0.008),
    "copper": (0.034, 0.008),
    "iron_ore": (-0.004, 0.006),
}
