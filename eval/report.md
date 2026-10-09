# Evaluation report

> 各指标的含义与读法见仓库根目录的 `DATA_NOTES.md` §5。指标名称（`recall@5`、`MRR`、`faithfulness` …）是代码与 CI 中的标识符，刻意保留英文，以便与日志、`report.json` 逐字对应。

- generated: 2026-10-09T04:34:58.809118+00:00
- answerer: `extractive`  |  embedder: `hashing (512d)`
- index: 705 documents / 764 chunks (numpy)
- questions: 20  |  top_k: 5

## Summary

| metric | value |
| --- | --- |
| recall@5 | **1.000** |
| MRR | 0.938 |
| answer faithfulness | 1.000 |
| answer correctness | 0.833 |
| citation validity | 1.000 |
| refusal accuracy | 1.000 |

## Per question

| id | rank | faithful | correct | cited | mode | question |
| --- | --- | --- | --- | --- | --- | --- |
| `cn-re-quota` | 1 | 1.00 | 0.67 | 1.00 | extractive | 中国稀土集团 2026 年上半年稀土开采总量控制指标是多少？ |
| `cn-export-control` | 1 | 1.00 | 0.50 | 1.00 | extractive | 镓、锗相关物项出口管制从什么时候开始施行？需要办什么手续？ |
| `cn-lithium-standard` | 1 | 1.00 | 1.00 | 1.00 | extractive | 2026 版碳酸锂产品标准把电池级纯度要求提高到了多少？ |
| `cn-re-licence` | 1 | 1.00 | 1.00 | 1.00 | extractive | 稀土出口许可证的办理时限调整成了多久？原因是什么？ |
| `au-cml` | 1 | 1.00 | 1.00 | 1.00 | extractive | 澳大利亚关键矿产清单 2026 年更新新增和移除了哪些矿种？ |
| `au-lithium-consult` | 2 | 1.00 | 0.00 | 1.00 | extractive | 澳大利亚锂精矿出口许可条件征求意见的核心要求是什么？ |
| `au-re-package` | 1 | 1.00 | 1.00 | 1.00 | extractive | 澳大利亚为稀土加工提供了多大规模的资金支持？重点支持哪些元素？ |
| `au-nickel-review` | 1 | 1.00 | 1.00 | 1.00 | extractive | 澳大利亚镍市场评估发现多少比例的镍企业处于现金亏损？ |
| `news-pilbara` | 1 | 1.00 | 1.00 | 1.00 | extractive | Pilbara Minerals 把锂精矿产量指引上调到了多少？ |
| `news-lithium-destock` | 4 | 1.00 | 0.00 | 1.00 | extractive | 锂电去库存进展如何？下游库存降到什么水平？ |
| `news-copper-tc` | 1 | 1.00 | 1.00 | 1.00 | extractive | 中国铜冶炼厂为什么同意削减产量？ |
| `news-lme-copper` | 1 | 1.00 | 1.00 | 1.00 | extractive | LME 铜库存降到了什么水平？原因是什么？ |
| `news-indonesia-nickel` | 1 | 1.00 | 1.00 | 1.00 | extractive | 印度尼西亚 2027 年镍矿配额调整了多少？ |
| `news-iron-ore-port` | 1 | 1.00 | 1.00 | 1.00 | extractive | 中国港口铁矿石库存处于什么水平？ |
| `news-re-magnet` | 1 | 1.00 | 1.00 | 1.00 | extractive | 稀土磁材出口在 9 月恢复得怎么样？ |
| `news-zinc-tc` | 1 | 1.00 | 0.50 | 1.00 | extractive | 锌加工费为什么跌到多年低点？ |
| `price-lithium` | 1 | 1.00 | 1.00 | 1.00 | extractive | 近 60 个交易日碳酸锂价格是涨还是跌？跌幅多少？ |
| `price-copper` | 1 | 1.00 | 1.00 | 1.00 | extractive | 近 60 个交易日铜价走势如何？ |
| `price-nickel` | 1 | 1.00 | 1.00 | 1.00 | extractive | 近 60 个交易日镍价跌幅是多少？ |
| `price-iron-ore` | 1 | 1.00 | 1.00 | 1.00 | extractive | 近 60 个交易日铁矿石价格变化大吗？ |

## Refusals

- **OK** `refuse-platinum` — uncovered commodity; measured best hit scores 0.46, above several real answers (mode=refusal, retrieved=0)
- **OK** `refuse-uranium` — uncovered commodity (mode=refusal, retrieved=0)
