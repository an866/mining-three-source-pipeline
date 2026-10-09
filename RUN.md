# 运行手册（RUN）

**中文** | English: `RUN.en.md`

命令参考。下面所有内容都**无需 API key、可离线运行**；增强项集中在最后一节。

---

## 0. 安装

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"         # 需要可选后端时加 ".[chroma]"
```

需要 Python 3.11 或更高版本。`make install` 等价。

---

## 1. 建索引

```bash
python -m pipeline.ingest --days 30          # 约 4 秒
python -m pipeline.ingest --days 30 --json   # 机器可读报告
python -m pipeline.ingest --days 30 --no-save   # 只在内存构建，不落盘
make ingest
```

产出 `.index/`（`manifest.json`、`chunks.jsonl`、`vectors.npy`）。manifest 里记录了嵌入器与维度，
两者都会在**加载时**校验——过期索引会在**启动时**带着修复方法失败，而不是在**查询时**抛出一个
形状错误。

人类可读输出：

```
documents      : 705
chunks         : 764
duplicates     : 56 removed (collector_identity=12, near_duplicate=24, title=20)
window         : 30 days (offline)
index          : .index
per kind       : news=200, policy=200, price=305
chunk chars    : mean=464.6 min=144.0 max=898.0
```

只有当**某个源失败**且**语料跌破底线**时，退出码才非零，因此 CI 能区分"语料太薄"和"语料正常"。

要真正联网采集而不是使用生成语料：

```bash
python -m pipeline.ingest --days 7 --online    # 或 MINING_OFFLINE=false
```

联网来源：6 个新闻 RSS feed、中澳政府公告页面、LME/SHFE/Mysteel 接口。某个源失败只损失它自己的
文档并记入 `source_errors`，整轮运行仍会产出可用的索引。响应缓存在 `.cache/` 下，并采用
stale-while-error 策略，因此 TTL 内的 feed 抖动是不可见的。

---

## 2. 起服务

```bash
mining-serve                          # 127.0.0.1:8000
python -m serve.app --port 9000
python -m serve.app --reload          # 开发模式
make serve
```

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/` | GET | 服务信息与示例请求体 |
| `/health` | GET | `status`、`index_loaded`、文档计数；降级时含 `error` |
| `/stats` | GET | 各来源族的文档/chunk 数、嵌入器、后端 |
| `/query` | POST | 主接口（请求体见下） |
| `/reload` | POST | 重新采集索引后无需重启即可重载 |
| `/ingest` | POST | 除非 `MINING_ALLOW_INGEST_ENDPOINT=1`，否则禁用 |

`/query` 请求体：

```json
{ "question": "印度尼西亚 2027 年镍矿配额调整了多少？",
  "top_k": 5,
  "synthesise": true,
  "filters": { "kinds": ["news"], "commodity": "nickel" } }
```

- `top_k` 可选（默认取 `MINING_TOP_K`）。
- `synthesise: false` 即使在配置了 LLM key 的情况下也强制走抽取式作答。
- `filters` 可选；不传则由规划器自动推断。**实际生效的过滤条件会在响应里回显**，因此结果集偏少
  时可以解释原因，而不是靠猜。
- 未知字段返回 422，所以拼错（例如 `"topk"`）会**大声失败**，而不是静默地返回 5 条结果。

索引缺失**不会**让服务崩溃：服务照常启动，`/health` 报告 `index_loaded: false` 并给出原因，
`/query` 返回 503 并附上构建索引的命令。

交互式文档在 `/docs`。服务没有鉴权且只绑定回环地址——见 `pipeline/config.py` 里的 `serve_host`。

### 示例

```bash
# 跨语言：用中文问一篇英文电讯稿
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"印度尼西亚 2027 年镍矿配额调整了多少？"}' | python -m json.tool

# 价格问题：问 60 个交易日的区间走势
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"近 60 个交易日碳酸锂价格是涨还是跌？"}' | python -m json.tool

# 被拒答：因为语料没有收录铂金
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"铂金最近的价格走势如何？"}' | python -m json.tool

curl -s localhost:8000/health | python -m json.tool
curl -s localhost:8000/stats  | python -m json.tool
```

---

## 3. 跑评测

```bash
python -m eval.harness                 # 打印汇总，并写出 eval/report.{md,json}
python -m eval.harness --top-k 10
python -m eval.harness --json          # 逐题完整报告输出到 stdout
python -m eval.harness --no-synthesise # 强制走抽取式作答
make eval
```

20 道计分题 + 2 个拒答用例。当 recall@5 < 0.9 或 faithfulness < 0.95 时以非零码退出，因此可以
用来卡 CI。`--out DIR` 可改变报告输出目录。

---

## 4. 探针与测试

```bash
python -m scripts.probe_retrieval -k 15   # 16 道硬题，打印"正确 URL 的排名"
pytest                                     # 214 个测试，约 2 秒
pytest -m "not slow"                       # 跳过端到端 ingest
pytest tests/test_vectorstore.py -q

ruff check . && ruff format --check .
mypy                                       # strict 模式
make check                                 # lint + 类型 + 测试
```

`scripts/probe_retrieval.py` 是调检索时的**快速循环**：它会打印那些曾经失败过的问题中，期望文档
最终排在第几名。

---

## 5. Docker

```bash
docker compose up --build          # 构建、建索引、在 :8000 提供服务
docker build -t mining-pipeline .
docker run --rm -p 8000:8000 mining-pipeline
```

镜像在**构建期**就完成离线语料的入库，因此容器启动时索引已就绪，且不需要任何网络访问。

---

## 6. 可选增强

两者都是即插即用的，且都会被评测集量化，所以"它到底有没有变好"是有答案的。

```bash
# 真实嵌入
export MINING_EMBEDDER=openai
export LLM_EMBEDDING_API_KEY=sk-...            # 或 LLM_API_KEY
export LLM_EMBEDDING_MODEL=text-embedding-3-small
python -m pipeline.ingest --days 30            # 必须重建：维度变了

# 真实综合作答
export LLM_API_KEY=sk-...                      # provider 默认 DeepSeek
export LLM_MODEL=deepseek-chat
mining-serve

python -m eval.harness                         # 与离线基线对比
```

任何 OpenAI 兼容端点都能用：OpenAI、DeepSeek、Voyage、本地 vLLM 或 Ollama 网关。更换嵌入器后
**必须重新入库**——如果忘了，manifest 校验会告诉你。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MINING_OFFLINE` | `true` | 永不联网 |
| `MINING_EMBEDDER` | `hashing` | `hashing` 或 `openai` |
| `MINING_EMBEDDING_DIM` | `512` | 哈希嵌入向量维度 |
| `MINING_VECTOR_BACKEND` | `numpy` | `numpy` 或 `chroma` |
| `MINING_LEXICAL_WEIGHT` | *随嵌入器* | 固定混合权重（哈希 `0.6` / 学习型 `0.35`） |
| `MINING_TOP_K` | `5` | 每次查询默认返回命中数 |
| `MINING_INDEX_DIR` | `.index` | 索引位置 |
| `MINING_CACHE_DIR` | `.cache` | HTTP 缓存位置 |
| `MINING_SERVE_HOST` / `MINING_SERVE_PORT` | `127.0.0.1` / `8000` | 绑定地址 |
| `MINING_ALLOW_INGEST_ENDPOINT` | `false` | 启用 `POST /ingest` |
| `LLM_API_KEY`、`LLM_MODEL`、`LLM_BASE_URL` | — | 对话综合作答 |
| `LLM_EMBEDDING_API_KEY`、`LLM_EMBEDDING_MODEL`、`LLM_EMBEDDING_BASE_URL` | — | 嵌入 |

---

## 7. 故障排查

**`no index found in .index`** —— 运行 `python -m pipeline.ingest`。若服务已在运行，也可以
`POST /reload`。

**`index was built with N-dimensional vectors but the configured embedder produces M`** ——
你改了 `MINING_EMBEDDER` 或 `MINING_EMBEDDING_DIM` 但没有重新入库。重建索引即可。

**答案的 `mode` 始终是 `"extractive"`** —— 没有设置 `LLM_API_KEY`，这是默认行为，不是错误。
抽取式答案会直接引用检索到的证据原句。

**`POST /ingest` 返回 403** —— 这是设计如此。设置 `MINING_ALLOW_INGEST_ENDPOINT=1`，或直接运行
`python -m pipeline.ingest`。

**某个问题被拒答，回复 "This corpus carries no documents for: …"** —— 该问题点名了一个索引未
收录的品种。这是覆盖检查在正常工作，见 `pipeline/retrieval.uncovered_minerals`。
