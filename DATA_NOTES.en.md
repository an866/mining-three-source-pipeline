# Data notes

**English** | 中文: `DATA_NOTES.md`

What the corpus is, where it comes from, and every measurement that a design decision in
this repository rests on. The point of this file is that nothing here should have to be
taken on trust.

---

## 1. The corpus is generated, and that is stated plainly

**Every document in the default corpus is synthetic.** No collected market data, no
scraped article text, no real government notice is bundled with this repository. The
numbers are shaped like the real instruments — LME 3M copper in USD/t, SHFE lithium
carbonate in CNY/t, Mysteel 62% Fe CFR Qingdao, Australia's Critical Minerals List, China's
rare-earth total-control quotas — but they are fabricated.

Why the default is synthetic:

1. **Reproducibility.** The evaluation measures recall@5. If the corpus changed between
   runs, so would the metric, and a regression would be indistinguishable from a different
   news day. Price values are keyed on the calendar date and document selection is seeded,
   so *the same date always yields the same corpus*.
2. **Ground truth has to be true of the index.** An evaluation set written by hand against a
   corpus nobody controls measures nothing. Here the corpus is built *around* a fixed anchor
   table (§3), and the eval questions resolve their expected URLs through that table.
3. **A fresh clone runs with no key and no network.** The alternative is a demo that only
   the author can run.

The live path exists and is real: RSS feeds for news, government pages for policy, venue
APIs for prices. It is selected with `MINING_OFFLINE=false`. It is not the tested path,
because public feeds change without notice and a test suite that depends on them is a test
suite that fails on someone else's schedule.

### What the corpus contains

| source family | documents | window | what it is |
| --- | --- | --- | --- |
| news | 200 | 30 days | 8 anchored stories + deterministic filler, across 4 outlets |
| policy | 200 | 30 days | 4 Chinese + 4 Australian anchored instruments + filler notices |
| price | 305 | 91 days | 5 commodities × 60 sessions + 1 trend summary each |
| **total** | **705** | | after de-duplication; 764 chunks |

Floors: ≥ 200 items per source, ≥ 30-day window, ≥ 600 documents total. All three are
asserted in `tests/test_ingest.py`, not merely reported.

### Prices

Five benchmarks, sixty trading sessions each. Direction is **guaranteed**, not hoped for:
each series is pinned to a target start-to-end change and given bounded noise that starts
and ends at exactly zero, so the reported change survives the noise untouched.

| commodity | 60-session change | direction |
| --- | --- | --- |
| lithium (SHFE carbonate) | −18.50 % | down |
| nickel (LME 3M) | −9.20 % | down |
| zinc (LME 3M) | −4.50 % | down |
| copper (LME 3M) | +3.40 % | up |
| iron ore (Mysteel 62 % Fe) | −0.40 % | flat |

The ±0.50 % flat band exists because a 0.4 % drift over three months is noise, and calling
it a direction would make the direction questions unstable. The eval derives its expected
facts from this same table (`eval/ground_truth.price_key_facts`), so changing a target
changes the expectation automatically.

---

## 2. De-duplication: the threshold was measured twice

Mining news is syndicated aggressively, so a URL set catches almost none of the duplication
that matters. The cascade has three tiers — canonical-URL identity, normalised title within
a source family, 5-shingle Jaccard, all confirmed behind LSH banding.

**The measurement.** Similarity was computed across the generated corpus to find where the
decision boundary actually lies:

| pair type | 5-shingle Jaccard |
| --- | --- |
| the most similar pair of genuinely *different* documents (two long policy notices sharing a clause) | **0.86** |
| a reprint under a rewritten headline, body unchanged | **1.00** |

So the threshold is **0.90**: it has a real margin on both sides. The first attempt used
0.85 and produced **exactly one false merge** — two unrelated policy notices collapsed into
one, and the corpus lost a document. That is what the measurement was for. The consequence
is recorded here because a threshold without its evidence is just a magic number.

**Injected duplication and where it goes.** The generator plants 28 duplicates per source
(6 exact re-collections, 10 same-headline syndications, 12 rewritten headlines over verbatim
bodies) — 56 in total. All 56 are removed, and the ingest report now accounts for every one:

```json
"duplicates_removed": 56,
"duplicates_by_stage": {
  "collector_identity": 12,   // exact re-collections, caught by the source-level guard
  "title": 20,                // same headline, different outlet
  "near_duplicate": 24        // rewritten headline
}
```

The `collector_identity` line exists because of a reporting bug worth recording: the
collector's identity guard and the cascade's tier 1 use the same key, and the guard runs
first. Before the breakdown was added, `duplicates_removed` reported 44 while the run had
actually removed 56 — understating duplication by 21 %, with the missing 12 invisible. The
cascade's `exact_id` counter is therefore **0** on this corpus by construction, which is
documented in `pipeline/dedup.py` rather than left to look like dead code.

Reprints are generated only from *filler* documents. A reprint displacing an anchor would
change which URL holds an evaluated fact and silently break recall@5 — so the anchors are
kept clean and the evaluation is independent of the de-duplication behaviour being measured.

---

## 3. The anchor table

`pipeline/synthetic/anchors.py` holds 16 planted facts with fixed URLs and fixed bodies: 8
news, 4 Chinese policy, 4 Australian policy. They are the corpus's ground truth and the eval
set's source of expected URLs. Each is a specific quotable fact — a quota figure, a licence
condition, a treatment-charge collapse.

The eval set resolves expected URLs through `anchor_urls(...)` rather than pasting URL
strings, so an anchor that moves fails loudly at import instead of quietly scoring zero.

Question style is deliberately *not* a paraphrase of the anchor title. Questions are written
the way an analyst asks — Chinese, English, shorthand — because the interesting failures in
a bilingual corpus are cross-lingual and exact-term, and a ground truth that only asks
questions in the document's own words would never exercise either.

---

## 4. Retrieval measurements

### The pre-trim was deleting the right answers

The store pre-trims candidates by pure cosine before running the lexical pass, as an
optimisation. Measured cosine rank of the correct document for six cross-lingual questions:

| question | cosine rank (of 764) |
| --- | --- |
| 印度尼西亚 2027 年镍矿配额调整了多少？ | 664 |
| LME 铜库存降到什么水平？ | 532 |
| 中国铜冶炼厂加工费谈判结果怎样？ | 730 |
| 澳大利亚关键矿产清单 2026 年更新新增了哪些矿种？ | 701 |
| 锂电去库存进展如何？ | 618 |
| 中国港口铁矿石库存处于什么水平？ | 710 |

A Chinese question against an English document has nearly no surface overlap, so it ranks in
the hundreds. The default `candidate_k` was 50: **every one of those documents was discarded
before its concept and exact-term matches were ever scored.** The pre-trim now only engages
above 2 000 candidates, which is inert at this corpus size. This single change took the
16-question probe from 9/16 to 13/16 top-15.

### Lexical weight sweep

Top-5 hits out of 16 probe questions, by blend weight:

| `lexical_weight` | top-5 | top-3 | top-1 |
| --- | --- | --- | --- |
| 0.25 | 13 | 13 | 10 |
| 0.35 (old default) | 15 | 14 | 12 |
| **0.60 (new default, hashing)** | **16** | **15** | **13** |
| 0.70 | 16 | 16 | 14 |

The weight is now chosen from the embedder rather than fixed. A learned embedder carries
semantic signal the lexical half does not, so dense leads (`0.35`). The hashing embedder's
cosine *is* surface overlap, so the two halves measure the same thing at two granularities
and the precise one leads (`0.60`). The sweep is why.

### Filter inference: two bugs of the same shape

Both were a **topic word read as a scope constraint**:

- `"quota"` was a policy genre keyword, so *"Indonesia nickel ore quota 2027"* planned
  `kind=policy` and excluded the news article that answered it.
- `"跌"` and a bare `"价"` were price genre keywords, so *"锌加工费为何跌至多年低点"*
  planned `kind=price` and excluded the news story.

Separately, a jurisdiction filter used to exclude every news and price chunk, because those
carry no jurisdiction. Naming 中国 in a question — *"中国铜冶炼厂加工费"*, where 中国 is the
subject, not a request for government documents — deleted the wire copy reporting the
result. Jurisdiction now only excludes a chunk that has one.

After both fixes the probe is 16/16 in top-15 and 15/16 in top-5.

### The remaining retrieval limitation

Ranked 7th on the probe: *"澳大利亚镍市场评估中亏损企业占比多少？"* against *"Nickel market
review finds 60 % of Australian operations loss-making"*. Both carry the same concepts
(australia, nickel, review), and Chinese-language filler that shares surface n-grams with
the question edges it out. This is the hashing embedder's cross-lingual ceiling: below the
concept lexicon it has nothing to work with. It is a documented limitation, not a hidden
one, and it is the first thing a learned embedding model would fix.

---

## 5. Evaluation results and how to read them

`python -m eval.harness` on the offline corpus, hashing embedder, extractive answerer,
20 graded questions, `top_k=5`:

| metric | value |
| --- | --- |
| recall@5 | 1.000 (20/20) |
| MRR | 0.938 |
| answer faithfulness | 1.000 |
| answer correctness | 0.833 |
| citation validity | 1.000 |
| refusal accuracy | 1.000 (2/2) |

**Faithfulness = 1.000 is a construction, not an achievement.** The extractive answerer
quotes sentences from the retrieved chunks, so every sentence is supported by definition.
The metric is a regression guard on that path and a real measurement only on the LLM path,
which is exactly where a hallucinated figure would otherwise go unnoticed.

**Correctness = 0.833 is the answerer's ceiling, not retrieval's.** Recall is 1.000, so the
document holding every expected fact is always in the window. Raising the sentence cap from
4 to 8 changes the score not at all — it was tested — so the loss is sentence *selection*:
the question *"中国稀土集团 2026 年上半年稀土开采总量控制指标是多少？"* answers with 135 000
and 19 150 tonnes but not 172 000, which is in an adjacent sentence of the same document and
retrievable by following citation `[1]`. Similarly, *"需要办什么手续？"* cannot be bridged by
sentence overlap to *"应当向商务部申请许可"* — 手续 and 许可 share no term. That is a
semantic gap, and closing it is what the LLM path is for.

**Refusal scope.** Refusal is driven by a commodity coverage check, so it catches questions
about minerals the index does not carry (platinum, uranium). It does **not** catch a
question that names a covered commodity but asks for a fact that is absent — *"智利 2027 年
锂矿产量预测是多少？"* would still be answered from the nearest lithium evidence. The
refusal cases are deliberately limited to the ones the deterministic path can honour;
the rest is the LLM prompt's refusal clause, which is why that clause is in the prompt.

---

## 6. Bilingual handling

Half the policy corpus is Chinese and half is English, and users ask in whichever language
they think in. Three consequences:

**Normalisation must be Unicode-native.** Anything that strips non-ASCII deletes the entire
Chinese corpus. `normalize_for_match` uses NFKC and keeps CJK intact.

**Tokenisation must serve two different jobs.** Matching uses CJK unigrams *and* bigrams, so
a one-character query (锂) can hit. Scoring uses maximal CJK runs instead, because n-gram
expansion bloats a Chinese query: 印度尼西亚 is one word and one unit but nine tokens, so a
document sharing three of nine n-grams looked a third relevant. `tokenize` decides *whether*
text matches; `search_units` decides *how much*.

**A concept lexicon bridges the languages.** ~53 hand-written concepts with their Chinese and
English surface forms (稀土 ↔ rare earth, 配额 ↔ quota, 加工费 ↔ treatment charge). It is a
deliberate, auditable limitation rather than a hidden one: it covers the domain vocabulary it
was written for and degrades to same-language matching outside it.

Two subtleties, both found by measuring:

- **IDF must be indexed on the same vocabulary it is looked up on.** The document-frequency
  table was originally built from n-grams while scoring queried maximal runs, so nearly every
  lookup missed and every CJK unit silently collapsed to the same maximum weight — turning
  the rare-term signal the table exists for back into a constant.
- **`氧化铝` (alumina) must not be refused as `铝` (aluminium).** The out-of-corpus mineral
  check lists `aluminium`/`aluminum` but deliberately not a bare `铝`, which is a substring of
  a commodity the corpus does carry.

---

## 7. Known gaps

- **`Fig. 3` splits into a sentence.** The sentence splitter's rule is whitespace-based and
  there is no abbreviation list. The resulting four-character fragment is discarded by the
  answerer's minimum sentence length, so it never reaches a user — but the module comment
  used to claim the case was handled, and it was not. The comment now says so.
- **Live collection is untested at scale.** It is written and structured, not exercised.
- **Chroma is a second implementation, not a benchmarked one.** It exists to prove the
  retrieval interface is a real seam; the numpy backend is the tested one.
