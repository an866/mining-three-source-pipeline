"""Ad-hoc retrieval probe: where does the expected document rank?

The eval harness (``eval/harness.py``) measures recall@5 over the whole ground
truth. This script is the smaller, faster loop used while tuning the lexical
half: it prints the rank of one known-correct URL for a handful of hard
questions, so a scoring change can be judged in seconds instead of a minute.

    python -m scripts.probe_retrieval            # offline index at .index
    python -m scripts.probe_retrieval -k 20      # widen the window

It has no assertions and no exit code semantics -- it is a torch, not a test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.embedding import build_embedder, describe_embedder
from pipeline.ingest import load_index
from pipeline.retrieval import Retriever, dedupe_hits

#: ``(question, anchor key, kind note)``. The key is resolved through
#: ``pipeline.synthetic.anchors`` so the URL is never duplicated by hand.
CASES: tuple[tuple[str, str], ...] = (
    ("中国稀土集团 2026 年上半年稀土开采总量控制指标是多少？", "cn_rare_earth_quota_2026h1"),
    ("镓、锗相关物项出口管制从什么时候开始施行？", "cn_export_control_gallium_germanium"),
    ("电池级碳酸锂纯度标准 2026 版提高了多少？", "cn_lithium_processing_standard"),
    ("稀土出口许可证办理时限调整到多久？", "cn_rare_earth_export_licence_slowdown"),
    ("澳大利亚关键矿产清单 2026 年更新新增了哪些矿种？", "au_disr_critical_minerals_list_update"),
    ("澳大利亚锂精矿出口许可条件征求意见的核心要求是什么？", "au_lithium_export_licence_review"),
    ("澳大利亚稀土加工支持包提供了多少资金？", "au_rare_earth_support_package"),
    ("澳大利亚镍市场评估中亏损企业占比多少？", "au_nickel_review"),
    ("Pilbara Minerals 锂精矿产量指引上调到多少？", "news_pilbara_guidance"),
    ("锂电去库存进展如何？", "news_lithium_destocking"),
    ("中国铜冶炼厂加工费谈判结果怎样？", "news_copper_smelter_cuts"),
    ("LME 铜库存降到什么水平？", "news_lme_copper_inventory"),
    ("印度尼西亚 2027 年镍矿配额调整了多少？", "news_indonesia_nickel_quota"),
    ("中国港口铁矿石库存处于什么水平？", "news_iron_ore_port_inventory"),
    ("稀土磁材出口 9 月恢复情况如何？", "news_rare_earth_magnet_exports"),
    ("锌加工费为何跌至多年低点？", "news_zinc_tc_collapse"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path(".index"))
    parser.add_argument("-k", "--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    from pipeline.synthetic.anchors import ANCHORS

    urls = {anchor.key: anchor.url for anchor in ANCHORS}
    store, _ = load_index(index_dir=args.index_dir)
    embedder = build_embedder()
    retriever = Retriever(store, embedder)
    # ``stats()`` rather than the backend's own counter, so the probe works against
    # either store implementation.
    print(f"embedder={describe_embedder(embedder)} chunks={store.stats().chunks}\n")

    misses = 0
    for question, key in CASES:
        expected = urls[key]
        hits = dedupe_hits(
            retriever.retrieve(question, top_k=max(args.top_k, 20)), max_per_document=1
        )
        rank = next(
            (i + 1 for i, hit in enumerate(hits[: args.top_k]) if hit.chunk.url == expected),
            None,
        )
        if rank is None:
            misses += 1
        mark = "OK " if rank and rank <= 5 else "-- "
        top = hits[0]
        print(
            f"{mark}rank={rank!s:>4}  {question[:34]:<36} "
            f"top1={top.chunk.url.rsplit('/', 1)[-1][:44]}"
        )
    print(f"\n{len(CASES) - misses}/{len(CASES)} found in top {args.top_k}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
