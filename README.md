# mining-three-source-pipeline

**中文** | English: `README.en.md`

24 小时面试题：采集矿业新闻、关键矿产政策与大宗商品价格三源数据，完成清洗、去重、向量化入库，
提供自然语言问答接口并给出引用，最后用一套**可复现的评测集**量化整条链路。

**所选题目：题 #1**（三源聚合管线 + `/query` 接口 + 评测集）。其余两题未作答。

```
$ python -m pipeline.ingest --days 30          # 采集 -> 去重 -> 分块 -> 嵌入 -> 建索引
$ mining-serve                                 # POST /query
$ python -m eval.harness                       # recall@5 + faithfulness，20 道题
```

以上三条命令**完全离线运行，不需要 API key，不需要网络**。真实的 LLM 与嵌入模型是可选的
增强项，叠加在一个本来就能跑通的默认实现之上（见下文「配置」一节）。

---

## 实测结果

离线语料（705 文档 / 764 chunk）、哈希嵌入器、抽取式作答、20 道题取前 5：

| 指标 | 数值 | 含义 |
| --- | --- | --- |
| **recall@5** | **1.000**（20/20） | 期望文档始终落在前 5 名内 |
| MRR | 0.938 | 且通常排在第 1 |
| **answer faithfulness** | **1.000** | 答案的每一句都能在检索到的证据里找到支撑 |
| answer correctness | 0.833 | 答案覆盖了期望事实的比例 |
| citation validity | 1.000 | 每个 `[n]` 都指向真实存在的引用 |
| refusal accuracy | 1.000 | 语料无法回答的问题被拒答，而不是猜一个 |

用 `python -m eval.harness` 复现；它会写出 `eval/report.md` 与 `eval/report.json`。
数值跨次运行稳定，见下文「可复现性」一节。

**请诚实地读这些数字。** `recall@5 = 1.000` 是在**哈希**嵌入器上测得的——哈希嵌入器本质上
是"披着稠密外衣的词法检索器"。这是**下限而非上限**：换成多语言学习型嵌入模型，跨语言召回还会
进一步提高，而评测集正是用来证明这一点的工具。`correctness = 0.833` 是**抽取式作答**的句子
选择上限，**不是检索的问题**——recall 已是 1.000，说明每个需要的文档都已在窗口内；抽取式路径
只挑四句话，偶尔会漏掉文档里确实存在的第三个数字。这两条caveat 连同其背后的测量过程都记在
`DATA_NOTES.md` 里。

---

## 它做了什么

```
                 ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
  sources   ──▶  │  news (RSS)   │   │ policy        │   │ prices        │
                 │  6 feeds      │   │ CN + AU gov   │   │ LME/SHFE/     │
                 │               │   │               │   │ Mysteel       │
                 └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                         │                   │                   │
                         └─────────┬─────────┴─────────┬─────────┘
                                   ▼                   ▼
                        ┌──────────────────────────────────────┐
                        │ clean → 3-tier de-duplicate → chunk  │
                        └──────────────────┬───────────────────┘
                                           ▼
                        ┌──────────────────────────────────────┐
                        │ hybrid index: cosine + IDF lexical   │
                        └──────────────────┬───────────────────┘
                                           ▼
                       ┌────────────────────────────────────────┐
   question  ───────▶  │ plan filters → retrieve → answer+cited │  ──▶  JSON
                       └────────────────────────────────────────┘
```

**采集。** 三个适配器背后是同一套接口。每个都有真实的联网路径（新闻用 RSS、政策用政府网站、
价格用各交易所接口），以及一条确定性的生成式兜底路径，在离线或某个源失败时启用。某个源挂掉
只损失它自己的文档并记入 `source_errors`，**永远不会中断整轮运行**。

**清洗。** 剥离 HTML、解码实体、去掉样板文字与图注、规整空白、兼容 RSS 与 ISO 两种日期格式。
归一化是 **Unicode 原生**的：一个 ASCII-strip 会静默删掉整个中文半壁语料。

**去重。** 三层，从便宜到昂贵——规范化 URL 身份、同来源族内的规范化标题、最后是 5-shingle
Jaccard ≥ 0.90（藏在 LSH 分带预筛之后）。在这份语料上，级联加上采集器的身份守卫移除了
**注入的 56 条重复中的 56 条**（`duplicates_by_stage` 会给出分层明细）。阈值是**测出来的**
而非拍脑袋定的：确实不同的两份文档最相似的一对是 0.86，而"改写标题、正文照搬"的转载是 1.00。

**建索引。** 约 900 字符一块、150 字符重叠，并携带指向原文的字符偏移，因此引用可以就地回读。

**检索。** 混合策略：在内存矩阵上做精确余弦，与 IDF 加权词法分数线性融合。另有一个双语概念
词表（约 53 个概念：稀土 ↔ rare earth、配额 ↔ quota），把中文提问桥接到英文文档——这是单纯
的词元重叠检索器与哈希嵌入器都做不到的。

**作答。** 检索优先，然后走 LLM 综合或从命中 chunk 中挑最优句子拼出抽取式答案。抽取式路径是
默认路径且**不是占位桩**：它的每个字都是引用，所以 faithfulness 天然为 1.0。引用只在一处生成，
两条路径都无法凭空造出引用。

---

## 快速开始

需要 Python 3.11+，无需 key，无需联网。

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m pipeline.ingest --days 30              # 约 4 秒，产出 .index/
mining-serve                                     # http://127.0.0.1:8000

curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"印度尼西亚 2027 年镍矿配额调整了多少？"}' | python -m json.tool
```

```json
{
  "question": "印度尼西亚 2027 年镍矿配额调整了多少？",
  "answer": "Indonesia's energy and mineral resources ministry cut the 2027 nickel ore
             production quota to 240 million wet metric tonnes from 275 million, a cut of
             roughly 13%. [1]",
  "mode": "extractive",
  "citations": [{"index": 1, "kind": "news", "title": "Indonesia trims nickel ore quota
                 for 2027", "url": "https://www.mining.com/indonesia-trims-nickel-ore-quota-for-2027/", ...}],
  "filters": {"kinds": [], "commodity": "nickel"},
  "latency_ms": 41
}
```

接口清单：`GET /health`、`GET /stats`、`POST /query`、`POST /reload`、`POST /ingest`
（除非设 `MINING_ALLOW_INGEST_ENDPOINT=1`，否则禁用）。完整命令行见 `RUN.md`。

---

## 配置

全部通过环境变量驱动，前缀 `MINING_`；LLM 相关用 `LLM_`。默认值即离线路径。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MINING_OFFLINE` | `true` | 永不联网；使用生成的语料 |
| `MINING_EMBEDDER` | `hashing` | `hashing`（确定性、离线）或 `openai` |
| `MINING_VECTOR_BACKEND` | `numpy` | `numpy`（精确、无需额外服务）或 `chroma` |
| `MINING_LEXICAL_WEIGHT` | *随嵌入器* | 词法半边的融合权重；哈希取 `0.6`，学习型取 `0.35` |
| `MINING_TOP_K` | `5` | `/query` 返回的命中数 |
| `LLM_API_KEY` | *(未设置)* | 启用 LLM 综合作答；不设置则走抽取式 |

开启增强路径：

```bash
export MINING_EMBEDDER=openai LLM_EMBEDDING_API_KEY=sk-...   # 真实嵌入
export LLM_API_KEY=sk-...                                    # 真实综合作答
python -m pipeline.ingest --days 30 && python -m eval.harness
```

两个升级都是即插即用的。**"它到底有没有变好"由评测集回答，而不是靠感觉。**

---

## 值得争论的设计决策

以下是"另一个靠谱工程师完全可能反着做"的地方，以及本代码为何这样选。更详细的版本在
`docs/architecture.md`。

**词法权重取决于嵌入器。** 学习型嵌入模型以稠密相似度为主导，所以词法半边只用来打破平局
（`0.35`）。而确定性哈希嵌入器的"余弦"**本身就是**特征哈希后的表面重叠，因此在这条路径上，
稠密与词法只是同一信号的两个粒度，**更精确的那一半必须主导**（`0.6`）。在 16 题探针上实测，
这一改动把 top-5 命中从 15/16 提升到 16/16。

**候选数少于 2000 时，跳过纯余弦预裁剪。** 先按余弦裁掉候选再算词法，听起来是个合理的优化，
但在这份语料上它**正在删掉正确答案**：中文问题对英文电讯稿的余弦排名是 530–730 / 764，于是
`candidate_k = 50` 在概念与精确词元被评分之前就把它们丢掉了。而对全部 764 个 chunk 做词法打分
只要几毫秒。

**问一个语料里没有的品种，要拒答，而不是硬答。** "铂金（platinum）最近的价格走势如何" 对价格
序列的得分是 **0.46**——**高于**若干真正可答的问题——因为它与每一篇价格文档都共享"价格/走势"。
**没有任何相关性阈值能把这种情况和真问题分开**，所以作答路径会把问题里的品种与索引的
`available_commodities()` 比对，不在其中就拒答。

**过滤器只按"体裁"推断，绝不按"主题"推断。** "锌加工费为何跌至多年低点" 里含有"跌"字，它
曾导致规划出 `kind=price`，从而排除了真正回答该问题的新闻。**主题词就是主题词。** 同理，辖区
过滤器永远不会排除一个没有辖区字段的 chunk——否则"中国铜冶炼厂加工费"（此处"中国"是**主语**）
会把报道该结果的电讯稿整篇删掉。

**这些数字是测出来的，不是选出来的。** 去重阈值取 0.90 而非更常见的 0.85；分句器的缩写词行为
被记录而非修掉。两者都来自对语料的观察：0.85 时级联产生了**恰好一次误合并**；而分句器切出的
四字符片段永远到不了用户面前，因为作答器会丢弃过短的句子。测量过程与原始数值都在
`DATA_NOTES.md`。

---

## 目录结构

```
pipeline/            采集 → 清洗 → 去重 → 分块 → 嵌入 → 建索引
  collectors/        三个源适配器 + 零依赖的 RSS/Atom 解析器
  synthetic/         确定性离线语料与植入的锚点事实
  textutils.py       Unicode 原生归一化、CJK 分词、双语概念词表
  vectorstore.py     混合检索的 numpy 后端（另有可选 Chroma）
  retrieval.py       基于规则的查询规划与检索
  answer.py          抽取式与 LLM 作答、引用、拒答
serve/app.py         FastAPI：/query /health /stats /reload /ingest
eval/                ground truth（20 题 + 2 个拒答用例）、指标、harness
tests/               214 个测试
DATA_NOTES.md        数据来源说明与本文引用的每一项测量（中文）
RUN.md               命令参考（中文）
docs/architecture.md 设计理由与扩展性说明（中文）
```

文档为中英双语，**以中文为主**：`README.md` / `DATA_NOTES.md` / `RUN.md` /
`docs/architecture.md` 是中文主版本，对应的英文版本为 `*.en.md`。

---

## 可复现性

语料是确定性的（同一天 → 同一份语料），嵌入器是确定性的，分数并列时按 chunk id 全序打破，
因此两次运行 `eval.harness` 会得到**完全相同的指标**。`latency_ms` 是唯一非确定性的字段，它被
排除在所有聚合之外。

`python -m eval.harness` 在 recall@5 低于 0.9 或 faithfulness 低于 0.95 时以非零码退出，
因此 CI 卡的是**检索质量**，而不只是"测试有没有过"。

## 测试

```bash
pytest                      # 214 个测试，约 2 秒（其中 12 个 ingest 测试标记为 slow）
ruff check . && ruff format --check .
mypy                        # strict 模式，38 个源文件，零问题
```

测试的重心放在**确实出过错的行为**上：预裁剪删掉跨语言锚点、辖区过滤器删掉电讯稿、由方向词
推断出体裁过滤、`duplicates_removed` 少报 21%、只在查询期才暴露的过期索引维度，以及一个注释
比代码承诺得更多的分句器。

## 已知局限

- 价格是**生成**的序列，只是形状仿照真实合约，并非采集到的行情。`DATA_NOTES.md` 有长篇说明；
  这里的任何数字都不应被当作市场数据。
- 联网采集是尽力而为、且未在大规模下验证的：公共新闻 RSS 与政府页面会毫无预告地变化。
  **被测试覆盖的是离线路径。**
- 跨语言检索依赖一份手写概念词表。它在其覆盖的领域词汇内有效，超出后降级为同语言匹配。
  真正的解法是多语言嵌入模型。
- 服务没有鉴权且只绑定回环地址。这是演示，不是生产部署。

---

> **AI 工具说明**：本项目代码与文档由 **WorkBuddy 降智版**（AI 编程助手，降智/简化能力模式）辅助完成。
