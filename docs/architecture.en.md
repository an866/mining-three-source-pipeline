# Architecture

**English** | 中文: `architecture.md`

Rationale for the choices that are not self-evident from the code. `README.en.md` is the
overview; `DATA_NOTES.en.md` holds the measurements; this file explains *why* the seams are
where they are.

---

## Layering

```
config ────────────── settings, cache dir, index dir, model config
errors ────────────── typed taxonomy: PipelineError → Source/Parse/Offline/VectorStore/LLM
logging_setup ─────── structured logs to stderr
cache ─────────────── disk cache with TTL and stale-while-error
http ──────────────── retries, timeouts, the single offline guard
models ────────────── pydantic types shared by every layer

collectors/ ───────── one adapter per source family, one interface
normalize ─────────── records → Document; HTML/entity cleaning
synthetic/ ────────── the deterministic offline corpus and the anchor table
textutils ─────────── normalisation, CJK tokenisation, concept lexicon, lexical scoring
dedup ─────────────── the three-tier cascade
chunking ──────────── documents → citable chunks
embedding ─────────── hashing (default) or OpenAI-compatible, plus a disk cache
vectorstore ───────── the hybrid contract: numpy (default) or Chroma
retrieval ─────────── query planning, retrieval, window widening, hit de-duplication
answer ────────────── extractive or LLM synthesis, citations, refusal
ingest ────────────── the only place the whole pipeline runs
serve/app ─────────── HTTP
eval/ ─────────────── ground truth, metrics, harness
```

Dependencies point in one direction. `answer` knows about `retrieval` and `vectorstore`;
`retrieval` knows about `vectorstore` and `embedding`; nothing in `pipeline/` knows about
`serve/`. `eval/` imports `pipeline/` and is never imported by it — the harness measures the
serving pipeline rather than re-implementing it.

---

## Why a vector store at all, and why that one

**Hybrid, not pure vector.** Dense retrieval is weak on exactly the tokens a mining question
turns on: `62% Fe`, `NI 43-101`, `135,000 吨`, `A$1.2bn`, `碳酸锂`. Lexical overlap is strong
there and weak where dense retrieval is strong. The two are blended as
`(1 - w) · cosine + w · lexical + title_weight · title_match`.

**The blend weight is a property of the embedder, not of the task.** A learned model encodes
meaning, so dense similarity is the workhorse and lexical breaks ties (`w = 0.35`). The
default hashing embedder's cosine *is* feature-hashed surface overlap — so on that path the
two halves measure the same thing at two granularities, and the finer one (IDF-weighted,
exact terms, cross-lingual concepts) must lead (`w = 0.60`). Measurements in `DATA_NOTES.en.md`.

**Filter before scoring.** `kind`/`jurisdiction`/`commodity`/recency are applied as a boolean
mask on the matrix, so a question scoped to "近 7 天的政策" cannot have its top-5 padded with
price rows. Filtering after ranking is how you end up with four results when five were asked
for.

**But a filter must only exclude what it can judge.** A jurisdiction filter excludes only
chunks that *have* a jurisdiction, because news and price chunks do not — and a country named
in a question is usually the subject, not a request for that country's statutes.

**Why not Chroma by default.** At 600–900 documents and a few thousand chunks, brute-force
exact cosine takes single-digit milliseconds. An ANN index would add a dependency, a
persistence format and a recall caveat in exchange for speed nobody needs. `ChromaVectorStore`
exists so the interface is a real seam rather than decoration: if only one class implements a
Protocol, the Protocol is aspiration.

### Scaling note

The numpy backend is O(n) per query in both work and memory; at 764 chunks that is nothing.
The three things that break first, in order:

1. **The in-memory matrix.** 764 × 512 float32 is 1.6 MB. At 10 million chunks it is 20 GB,
   and the answer is an ANN index (HNSW via `hnswlib`, or Chroma/pgvector) behind the same
   `VectorStore` protocol. Nothing else changes: `build_store` is the seam.
2. **The Python lexical pass.** `NumpyVectorStore.search` scores candidates in a loop, which
   is why the pre-trim exists. Above ~50 k chunks the lexical half should be inverted-indexed
   (a real BM25 implementation) rather than computed per candidate.
3. **`_df` rebuild on load.** One pass over the corpus today; at scale it becomes a
   persisted, versioned artifact — and then a thing that can disagree with the chunks it
   describes, which is why it is not persisted now.

---

## The de-duplication cascade

Cheapest tier first, because on a 749-document corpus the expensive tier would otherwise run
against every pair (≈280 000 comparisons) instead of the few thousand that LSH banding
produces.

| tier | key | catches |
| --- | --- | --- |
| 1 | `Document.id` (canonical URL + title + date) | exact re-collection |
| 2 | normalised title, within one source family | cross-domain syndication |
| 3 | 5-shingle Jaccard ≥ 0.90, via LSH banding | rewritten headline |

Three decisions inside it:

**Banding is strided, not prefix-based.** A prefix band only ever hashes the same shingles, so
a document whose edit touched a low-sorted shingle could fail to collide with its own reprint.
The pre-filter would silently become a source of false negatives instead of an optimisation.

**Reconciliation keeps the *materially* longer copy** (a 20 % band). Without the band, a
syndicated copy that merely appends a byline displaces the primary publication and every
citation points at an aggregator. Ties go to the incumbent, which keeps the outcome
independent of collection order.

**Tiers 1 and 2 are corpus-wide; tier 3 is scoped to a source family.** A policy notice and a
news item can legitimately share boilerplate, and merging them would delete a primary source
in favour of a secondary one.

The cascade's tier 1 shares its key with the collector's own identity guard, which runs first
and exists for a different reason (keeping the per-source count honest before the floor
check). The run's `duplicates_by_stage` reports both, because a completeness metric that
silently omits a stage is worse than no metric.

---

## Query planning: rules, not a model

`plan_filters` extracts recency, source family, jurisdiction and commodity with about eight
readable rules. A learned intent classifier here would add latency to every query and make a
wrong filter impossible to explain. The rules are wrong sometimes; when they were, it was a
one-line fix and the reasoning is in the git history and in `DATA_NOTES.en.md`:

- a *topic* word is not a *genre* constraint (`"quota"`, `"跌"`),
- a *country* is usually a subject, not a scope,
- an unset field means "unconstrained", never "match nothing".

The planner's output is echoed in the API response, so a user can see exactly which
constraints were applied.

**Window-widening is the one relaxation.** A hard "近 7 天" cut can return nothing when the
window is genuinely quiet, and an empty answer is worse than an answer from nine days ago
that says how old it is. So a filtered search that under-fills is retried with the window
multiplied by four, and citations carry their real dates. Jurisdiction and source family are
*not* relaxed: those are semantic constraints, and ignoring them produces answers about the
wrong thing.

---

## The coverage check

"铂金 (platinum) 最近的价格走势如何" scored 0.46 against the price series — higher than
several genuinely answerable questions — because it shares 价格/走势 with every price
document. No relevance threshold separates that from a real question, so the answer path
compares the question's commodity against `store.available_commodities()` and refuses when
the mineral is absent.

Reading `available_commodities` from the *index* rather than from a static list matters: after
a run where prices failed, a nickel question should be refused rather than answered from
nickel news.

This catches commodity gaps, not fact gaps — see the scope note in `DATA_NOTES.en.md` §5.

---

## Answering, and what faithfulness can and cannot tell you

Two answerers behind one interface. The extractive one is the default and is not a stub:
every word is a quote from an indexed document, so faithfulness is 1.0 by construction and
the metric is a regression guard on that path. The LLM path produces better prose and a weaker
guarantee, which is exactly what the harness measures.

`build_citations` is the only thing that creates `Citation` objects, and both paths draw from
the same `hits` list, so neither can invent a citation.

The system prompt is strict about three things a mining analyst would be angry about: figures
must be quoted with units and dates, policy must be distinguished from news from price data,
and an unsupported answer must be **refused rather than guessed**. Without the refusal clause,
a model asked about a commodity the corpus does not cover answers from general knowledge, and
the faithfulness metric measures the training data.

**What the metrics mean.** `recall@k` is a retrieval property and says nothing about prose.
`faithfulness` is a prose property and says nothing about retrieval. `correctness` separates
"retrieved the wrong document" from "quoted the right document badly" — a question can be
recalled, faithfully quoted, and still wrong. Reporting one number would hide that.

---

## Serving

**Sync `def` endpoints, not `async def`.** Retrieval is CPU-bound NumPy and the embedding
cache is blocking file I/O; declaring the handler `async` would run that on the event loop and
stall every other request. Sync handlers run in Starlette's threadpool, and the store is
read-only after load, so concurrent `/query` calls are safe without a lock. The only mutable
state is the engine reference, and `/reload` swaps it under a lock.

**A missing index is not a crash.** The service starts, `/health` says why it cannot serve,
and `/query` returns 503 with the remedy. Failing at import time gives a container that
crash-loops instead of one whose health check tells you what to run.

**Errors are typed.** A `PipelineError` becomes a 502 carrying its own message, not a bare
500. An LLM outage degrades to the extractive answer and labels itself `mode: "extractive"`,
so a caller can tell a genuinely extractive deployment from a degraded one.

**No authentication, bound to loopback.** Stated in the config and the README rather than
discovered in production. `/ingest` is off by default: a read API that can be told to rewrite
its own index is a surprising thing to expose.

---

## Deliberate omissions

- **No agent framework.** The pipeline is a straight line; wrapping it in one would add a
  dependency and a layer of indirection over four function calls.
- **No database.** The index is three files. Persisting chunk metadata in Postgres would add
  an operational dependency to a service whose whole point is that a fresh clone runs.
- **No re-ranker.** A cross-encoder would probably lift recall, and would also mean the
  offline default cannot run without a model download. The `Retriever` seam is where it goes.
- **No async collection.** Three adapters, sequential, each with its own timeout budget. Total
  wall clock offline is under a second and under ten seconds live; concurrency would buy
  seconds and cost a great deal of debuggability.
