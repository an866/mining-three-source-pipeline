# Run it

**English** | 中文: `RUN.md`

Command reference. Everything below works offline with no API key; the last section covers
the optional upgrades.

---

## 0. Install

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"         # add ".[chroma]" for the optional backend
```

Python 3.11 or newer. `make install` does the same thing.

---

## 1. Build the index

```bash
python -m pipeline.ingest --days 30          # ~4 s
python -m pipeline.ingest --days 30 --json   # machine-readable report
python -m pipeline.ingest --days 30 --no-save   # build and discard
make ingest
```

Writes `.index/` (`manifest.json`, `chunks.jsonl`, `vectors.npy`). The manifest records the
embedder and dimension, and both are verified on load — a stale index fails at startup with
a remedy rather than at query time with a shape error.

Human-readable output:

```
documents      : 705
chunks         : 764
duplicates     : 56 removed (collector_identity=12, near_duplicate=24, title=20)
window         : 30 days (offline)
index          : .index
per kind       : news=200, policy=200, price=305
chunk chars    : mean=464.6 min=144.0 max=898.0
```

Exit code is non-zero only when a source failed **and** the corpus fell below the floor, so
CI can tell "the corpus is thin" from "the corpus is fine".

To collect for real instead of using the generated corpus:

```bash
python -m pipeline.ingest --days 7 --online    # or MINING_OFFLINE=false
```

Live sources: 6 news RSS feeds, CN/AU government publication pages, LME/SHFE/Mysteel
endpoints. A failing source costs its own documents and is recorded in `source_errors`; the
run still produces a usable index. Responses are cached under `.cache/` with a
stale-while-error policy, so a feed outage inside the TTL is invisible.

---

## 2. Serve

```bash
mining-serve                          # 127.0.0.1:8000
python -m serve.app --port 9000
python -m serve.app --reload          # development
make serve
```

| endpoint | method | notes |
| --- | --- | --- |
| `/` | GET | service info and an example payload |
| `/health` | GET | `status`, `index_loaded`, counts, and `error` when degraded |
| `/stats` | GET | documents/chunks per source family, embedder, backend |
| `/query` | POST | the main endpoint (body below) |
| `/reload` | POST | re-read the index after a re-ingest, without a restart |
| `/ingest` | POST | disabled unless `MINING_ALLOW_INGEST_ENDPOINT=1` |

`/query` body:

```json
{ "question": "印度尼西亚 2027 年镍矿配额调整了多少？",
  "top_k": 5,
  "synthesise": true,
  "filters": { "kinds": ["news"], "commodity": "nickel" } }
```

- `top_k` is optional (default `MINING_TOP_K`).
- `synthesise: false` forces the extractive answer even when an LLM key is configured.
- `filters` is optional; omit it and the planner infers them. Whatever was applied is echoed
  in the response, so a small result set can be explained rather than guessed at.
- Unknown fields are rejected with 422, so a typo (`"topk"`) fails loudly instead of
  silently returning five results.

A missing index does not crash the service: it starts, `/health` reports
`index_loaded: false` with the reason, and `/query` answers 503 carrying the ingest command.

Interactive docs at `/docs`. The service has no authentication and binds to loopback — see
`serve_host` in `pipeline/config.py`.

### Examples

```bash
# cross-lingual: a Chinese question against an English wire story
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"印度尼西亚 2027 年镍矿配额调整了多少？"}' | python -m json.tool

# a price question over the 60-session series
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"近 60 个交易日碳酸锂价格是涨还是跌？"}' | python -m json.tool

# refused, because the corpus carries no platinum
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"铂金最近的价格走势如何？"}' | python -m json.tool

curl -s localhost:8000/health | python -m json.tool
curl -s localhost:8000/stats  | python -m json.tool
```

---

## 3. Evaluate

```bash
python -m eval.harness                 # prints the summary, writes eval/report.{md,json}
python -m eval.harness --top-k 10
python -m eval.harness --json          # full per-question report to stdout
python -m eval.harness --no-synthesise # force the extractive answerer
make eval
```

20 graded questions plus 2 refusal cases. Exits non-zero when recall@5 < 0.9 or faithfulness
< 0.95, so it can gate CI. `--out DIR` changes where the report is written.

---

## 4. Probes and tests

```bash
python -m scripts.probe_retrieval -k 15   # rank of the known-correct URL, 16 hard questions
pytest                                     # 214 tests, ~2 s
pytest -m "not slow"                       # skip the end-to-end ingest
pytest tests/test_vectorstore.py -q

ruff check . && ruff format --check .
mypy                                       # strict
make check                                 # lint + types + tests
```

`scripts/probe_retrieval.py` is the fast loop used while tuning retrieval: it prints where
the expected document ranked for questions that were once failing.

---

## 5. Docker

```bash
docker compose up --build          # builds, ingests, serves on :8000
docker build -t mining-pipeline .
docker run --rm -p 8000:8000 mining-pipeline
```

The image ingests the offline corpus at build time, so the container starts with a working
index and no network access.

---

## 6. Optional upgrades

Both are drop-in and both are measured by the harness, so the question "did it help?" has an
answer.

```bash
# real embeddings
export MINING_EMBEDDER=openai
export LLM_EMBEDDING_API_KEY=sk-...            # or LLM_API_KEY
export LLM_EMBEDDING_MODEL=text-embedding-3-small
python -m pipeline.ingest --days 30            # rebuild: the dimension changes

# real answer synthesis
export LLM_API_KEY=sk-...                      # provider defaults to DeepSeek
export LLM_MODEL=deepseek-chat
mining-serve

python -m eval.harness                         # compare against the offline baseline
```

Any OpenAI-compatible endpoint works: OpenAI, DeepSeek, Voyage, a local vLLM or an Ollama
gateway. Changing the embedder requires re-ingesting — the manifest check will tell you if
you forget.

| variable | default | effect |
| --- | --- | --- |
| `MINING_OFFLINE` | `true` | never touch the network |
| `MINING_EMBEDDER` | `hashing` | `hashing` or `openai` |
| `MINING_EMBEDDING_DIM` | `512` | hashing vector size |
| `MINING_VECTOR_BACKEND` | `numpy` | `numpy` or `chroma` |
| `MINING_LEXICAL_WEIGHT` | *by embedder* | pin the hybrid blend (`0.6` hashing / `0.35` learned) |
| `MINING_TOP_K` | `5` | default hits per query |
| `MINING_INDEX_DIR` | `.index` | index location |
| `MINING_CACHE_DIR` | `.cache` | HTTP cache location |
| `MINING_SERVE_HOST` / `MINING_SERVE_PORT` | `127.0.0.1` / `8000` | bind address |
| `MINING_ALLOW_INGEST_ENDPOINT` | `false` | enable `POST /ingest` |
| `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | — | chat synthesis |
| `LLM_EMBEDDING_API_KEY`, `LLM_EMBEDDING_MODEL`, `LLM_EMBEDDING_BASE_URL` | — | embeddings |

---

## 7. Troubleshooting

**`no index found in .index`** — run `python -m pipeline.ingest`. Or `POST /reload` if the
service is already up.

**`index was built with N-dimensional vectors but the configured embedder produces M`** — you
changed `MINING_EMBEDDER` or `MINING_EMBEDDING_DIM` without re-ingesting. Rebuild the index.

**Answers are always `mode: "extractive"`** — no `LLM_API_KEY` is set, which is the default
and is not an error. The extractive answer quotes the retrieved evidence.

**`POST /ingest` returns 403** — by design. Set `MINING_ALLOW_INGEST_ENDPOINT=1`, or run
`python -m pipeline.ingest`.

**A question is refused with "This corpus carries no documents for: …"** — the question names
a mineral the index does not carry. That is the coverage check working; see
`pipeline/retrieval.uncovered_minerals`.
