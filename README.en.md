# mining-three-source-pipeline

**English** | 中文: `README.md`

A 24-hour take-home: collect mining news, critical-minerals policy and commodity prices;
clean, de-duplicate and index them into a vector store; answer natural-language questions
over the result with citations; and measure the whole thing with a reproducible
evaluation harness.

**Question chosen: 题 #1** (three-source aggregation pipeline + `/query` API + eval set).
The other two questions were not attempted.

```
$ python -m pipeline.ingest --days 30          # collect -> dedup -> chunk -> embed -> index
$ mining-serve                                 # POST /query
$ python -m eval.harness                       # recall@5 + faithfulness, 20 questions
```

Everything above runs **offline, with no API key and no network**. A real LLM and a real
embedding model are optional upgrades on top of a working default (see the
**Configuration** section below).

---

## Results

Measured on the offline corpus (705 documents / 764 chunks), hashing embedder, extractive
answerer, 20 graded questions in 5 slots:

| metric | value | what it means |
| --- | --- | --- |
| **recall@5** | **1.000** (20/20) | the expected document is always in the top 5 |
| MRR | 0.938 | and it is usually first |
| **answer faithfulness** | **1.000** | every sentence of the answer is supported by retrievable evidence |
| answer correctness | 0.833 | fraction of expected facts present in the answer |
| citation validity | 1.000 | every `[n]` addresses a real citation |
| refusal accuracy | 1.000 | questions outside the corpus are refused, not guessed at |

Reproduce with `python -m eval.harness`; it writes `eval/report.md` and `eval/report.json`.
The numbers are stable across runs — see the **Reproducibility** section below.

**Reading these honestly.** `recall@5 = 1.000` is measured against a *hashing* embedder,
which is a lexical retriever wearing a dense hat. That is the floor, not the ceiling: a
learned multilingual embedder would raise cross-lingual recall further, and the harness is
the thing that would show it. `correctness = 0.833` is a ceiling of the *extractive*
answerer, not of retrieval — recall is 1.000, so every needed document is already in the
window; the extractive path picks four sentences and sometimes omits a third figure that
the document does contain. `DATA_NOTES.en.md` records both caveats with the measurements
behind them.

---

## What it does

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

**Collect.** Three adapters behind one interface. Each has a real live path (RSS for news,
government pages for policy, venue APIs for prices) and a deterministic generated fallback
used when offline or when a source fails. A dead source costs its own documents and lands
in `source_errors`; it never aborts the run.

**Clean.** HTML stripped, entities decoded, boilerplate and captions removed, whitespace
normalised, dates parsed with both RSS and ISO formats. Normalisation is Unicode-native: an
ASCII-strip would silently delete the entire Chinese half of the corpus.

**De-duplicate.** Three tiers, cheapest first — canonical-URL identity, normalised title
within a source family, then 5-shingle Jaccard ≥ 0.90 confirmed behind LSH banding. On this
corpus the cascade plus the collector's identity guard remove 56 of 56 injected duplicates
(`duplicates_by_stage` in the ingest report shows the split). The threshold was *measured*,
not chosen: the most similar pair of genuinely different documents sits at 0.86 and a
reprint under a rewritten headline sits at 1.00.

**Index.** Chunks of ~900 characters with 150-character overlap, carrying character offsets
into the original document so a citation can be re-read in place.

**Retrieve.** Hybrid: exact cosine over an in-memory matrix, blended with an IDF-weighted
lexical score. Plus a bilingual concept lexicon (~53 concepts: 稀土 ↔ rare earth, 配额 ↔
quota) that bridges Chinese questions to English documents, which neither a token-overlap
retriever nor a hashing embedder can do on its own.

**Answer.** Retrieval-first, then either an LLM synthesis or an extractive answer assembled
from the best sentences of the retrieved chunks. The extractive path is the default and is
not a stub: every word of it is a quote, which is why faithfulness is 1.0 by construction.
Citations are built in one place, so neither path can invent one.

---

## Quickstart

Requires Python 3.11+. No key, no network.

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m pipeline.ingest --days 30              # ~4 s, writes .index/
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

Endpoints: `GET /health`, `GET /stats`, `POST /query`, `POST /reload`, `POST /ingest`
(disabled unless `MINING_ALLOW_INGEST_ENDPOINT=1`). `RUN.en.md` has the full command surface.

---

## Configuration

Everything is environment-driven with the `MINING_` prefix; the LLM settings use `LLM_`.
Defaults are the offline path.

| variable | default | effect |
| --- | --- | --- |
| `MINING_OFFLINE` | `true` | never touch the network; serve the generated corpus |
| `MINING_EMBEDDER` | `hashing` | `hashing` (deterministic, offline) or `openai` |
| `MINING_VECTOR_BACKEND` | `numpy` | `numpy` (exact, no service) or `chroma` |
| `MINING_LEXICAL_WEIGHT` | *from embedder* | blend weight of the lexical half; `0.6` for hashing, `0.35` for a learned model |
| `MINING_TOP_K` | `5` | hits returned by `/query` |
| `LLM_API_KEY` | *(unset)* | enables LLM synthesis; without it answers are extractive |

Turning on the rich path:

```bash
export MINING_EMBEDDER=openai LLM_EMBEDDING_API_KEY=sk-...   # real embeddings
export LLM_API_KEY=sk-...                                    # real synthesis
python -m pipeline.ingest --days 30 && python -m eval.harness
```

Both upgrades are drop-in. The eval harness is what tells you whether they helped.

---

## Design decisions worth arguing about

These are the choices where a reasonable engineer might have gone the other way, and why
this code went the way it did. Longer versions are in `docs/architecture.en.md`.

**The lexical weight depends on the embedder.** A learned embedding model leads with dense
similarity, so the lexical half only breaks ties (`0.35`). The deterministic hashing
embedder's "cosine" *is* feature-hashed surface overlap, so on that path dense and lexical
measure the same thing at two granularities and the precise half must lead (`0.6`). Measured
on the 16-question probe, that change took top-5 recall from 15/16 to 16/16.

**The pure-cosine pre-trim is skipped below 2 000 candidates.** Trimming candidates by
cosine before scoring them lexically is a sensible optimisation that was, on this corpus,
deleting the right answers: a Chinese question against an English wire story ranks 530–730
of 764 by cosine, so a `candidate_k` of 50 removed it before its concept and exact-term
matches were ever scored. Lexical scoring all 764 chunks costs milliseconds.

**A question about a mineral the corpus does not carry is refused, not answered.** "铂金
(platinum) 最近的价格走势如何" scored 0.46 against the price series — *above* several
genuinely answerable questions — because it shares 价格/走势 with every price document. No
relevance threshold can separate that from a real question, so the answer path checks the
question's commodity against the index's `available_commodities()` and refuses when it is
absent.

**Filter inference only fires on genre, never on topic.** "锌加工费为何跌至多年低点"
contains the word "跌", which once planned `kind=price` and excluded the news story that
answered it. A topic word is a topic. Likewise a jurisdiction filter never excludes a chunk
that carries no jurisdiction, because otherwise "中国铜冶炼厂加工费" — where 中国 is the
subject — deletes the wire copy reporting the result.

**Numbers that were measured, not chosen.** The de-duplication threshold is 0.90 rather than
the more usual 0.85, and the sentence splitter's abbreviation behaviour is documented rather
than fixed. Both came from looking at the corpus: at 0.85 the cascade produced exactly one
false merge, and the splitter's four-character fragments never reach a user because the
answerer drops short sentences. Measurements and their raw values are in `DATA_NOTES.en.md`.

---

## Repository layout

```
pipeline/            collect → clean → dedup → chunk → embed → index
  collectors/        three source adapters + a dependency-free RSS/Atom parser
  synthetic/         the deterministic offline corpus and the planted anchors
  textutils.py       Unicode-native normalisation, CJK tokenisation, concept lexicon
  vectorstore.py     hybrid numpy backend (+ optional Chroma)
  retrieval.py       rule-based query planning and retrieval
  answer.py          extractive and LLM answer synthesis, citations, refusal
serve/app.py         FastAPI: /query /health /stats /reload /ingest
eval/                ground truth (20 questions + 2 refusals), metrics, harness
tests/               214 tests
DATA_NOTES.md        data provenance and every measurement quoted above (中文)
DATA_NOTES.en.md     the English version of the same notes
RUN.md               command reference (中文)
RUN.en.md            the English version
docs/architecture.md design rationale and scaling notes (中文)
docs/architecture.en.md  the English version
```

Documentation is bilingual with Chinese as the primary language; `README.md` is the
Chinese entry point and this file is its English counterpart.

---

## Reproducibility

The corpus is deterministic (same date → same documents), the embedder is deterministic,
and score ties break on chunk id, so two runs of `eval.harness` produce identical metrics.
`latency_ms` is the one non-deterministic field and it is excluded from every aggregate.

`python -m eval.harness` exits non-zero if recall@5 drops below 0.9 or faithfulness below
0.95, so CI gates on retrieval quality rather than only on tests passing.

## Testing

```bash
pytest                      # 214 tests, ~2 s (the 12 ingest tests are marked `slow`)
ruff check . && ruff format --check .
mypy                        # strict, 38 source files, clean
```

The suite is weighted towards behaviour that was actually wrong at some point: the pre-trim
deleting cross-lingual anchors, a jurisdiction filter deleting wire copy, a genre filter
inferred from a direction word, `duplicates_removed` under-reporting by 21%, a stale index
dimension that only failed at query time, and a sentence splitter whose comment claimed more
than the code did.

## Limitations

- Prices are a **generated** series shaped like the real contracts, not collected quotes.
  `DATA_NOTES.en.md` says so at length; no number here should be mistaken for market data.
- Live collection is best-effort and untested at scale: public news RSS and government pages
  change without notice. The offline path is the tested one.
- Cross-lingual retrieval rests on a hand-written concept lexicon. It works for the domain
  vocabulary it covers and degrades to same-language matching outside it. A multilingual
  embedding model is the real answer.
- The service has no authentication and binds to loopback. It is a demo, not a deployment.

---

> **Tooling note**: this project's code and docs were produced with **WorkBuddy 降智版**
> (an AI coding assistant running in reduced-capability mode).
