# Case study: when vector compression does not help

**A real check of TurboQuant against two similarity features of a legal-documents platform. Verdict for both: it does not help.**

This chapter complements the technical guide. Sections 4 and 5 there show where TurboQuant shines: the KV
cache and large vector indexes. This chapter shows the opposite case, which a workshop needs just as much:
how to tell, **before** compressing anything, whether vector compression can move the problem you have.
The code is in `es_bench/` in this repository and runs on a laptop.

> **In plain words.** We had two slow or costly "find similar items" features and a new compression technique.
> Before benchmarking compression, we asked three questions: where do the vectors live, what score does the
> product actually show, and what is really slow? For one feature the answers ruled compression out
> completely. For the other they showed it is not the first thing to fix.

---

## 1. The two features

| | A. Similar provisions | B. Suggested responses for comment memos |
|---|---|---|
| What the user sees | "Provisions similar to this one, above X %" across a firm's whole provision database | Past responses to similar questions, while answering a new comment |
| Score of record | **Edit-distance similarity** (`rapidfuzz.fuzz.ratio`, 0–100) on the cleaned text | **Cosine** between OpenAI embeddings (3,072 dimensions) |
| Where it lives | A precomputed pair matrix in PostgreSQL; the score is stored if ≥ 30, and the UI defaults to 70 | A vector database collection; search is **exact** (brute force), filtered by firm |
| Pain | Storage blow-ups and slow writes: every new provision is compared with every other one in the firm | Latency per request (embedding call, vector search, re-ranker, LLM relevance check) |
| Vectors involved? | Embeddings exist in the search index, but **no feature reads them** | Yes, they are the search |

---

## 2. Question 1: where do the vectors live?

The first benchmark plan assumed a vector-database collection of provision embeddings with a cosine
auto-merge threshold. Reading the code showed that neither exists. Provision embeddings are written to the
**Elasticsearch** index (`text-embedding-3-small`, 1,536 dimensions), and the only feature that read them,
a semantic search mode, had been removed months earlier. Both "similar provisions" and auto-merging score
pairs with edit distance.

**Lesson.** A benchmark is only as good as its premise. Five minutes of reading where vectors are written
and read saved a benchmark against a collection that is not there. The benchmark was rewritten for
Elasticsearch (`es_bench/`).

---

## 3. The experiment (`es_bench/`)

**Data.** 2,501 distinct real provision texts, cleaned exactly as the platform cleans them, embedded with
the platform's model and indexed in a local Elasticsearch 8.18 with the platform's mapping.

**Ground truth.** The score the product actually uses: exact `fuzz.ratio` of 500 provisions against all
2,501.

**Question.** If we used cosine neighbours (float32 or compressed) or trigram candidates to choose *which*
pairs to score, instead of scoring every pair, how many of the real pairs would we find?

**Validation first.** 13 known-answer checks run before any number is trusted (`canary.py`). Examples:
the trigram code matches PostgreSQL's `pg_trgm` to 5·10⁻⁹, an exact Elasticsearch index reproduces
exact top-k, and the sample loader returns exactly the expected rows. They caught two bugs in the
benchmark itself before it produced a single result.

### 3.1 How many pairs are "similar"? The number that decides everything

| Stored if fuzz.ratio ≥ | 30 | 40 | 50 | 60 | 70 |
|---|---|---|---|---|---|
| Share of all pairs kept | **70 %** | 45 % | **1.6 %** | 0.8 % | 0.7 % |

Two *unrelated* long legal provisions already score about 38, because edit distance on long English text
has a high baseline. So at a floor of 30 the "sparse" similarity matrix is in fact almost **dense**: for a
firm of 500,000 provisions that is about 87 billion pairs. At 50 it is about 2 billion.

### 3.2 Can a candidate filter find the real pairs?

Share of the pairs with fuzz.ratio ≥ 70 (the UI default) found among each provision's candidates:

| Candidates per provision | 10 | 50 | 100 | 200 |
|---|---|---|---|---|
| Cosine neighbours, float32 | 20 % | 54 % | 82 % | 99.8 % |
| Cosine neighbours, TurboQuant 2-bit | 20 % | 54 % | 82 % | 99.8 % |
| Trigram neighbours (`pg_trgm`) | 20 % | 55 % | 82 % | **100 %** |

At a floor of 30 every filter fails, because the "neighbours" are 70 % of the firm.

### 3.3 Does compression change anything?

| Method | Memory vs float32 | Top-10 overlap with float32 (no rescore → rescore 2×) | Recall of fuzz ≥ 70 pairs @200 |
|---|---|---|---|
| float32 | 1× | 1.000 | 99.8 % |
| scalar int8 | 4× smaller | 0.949 → 0.999 | 99.8 % |
| binary 1-bit | 32× smaller | 0.824 → 0.962 | 99.7 % |
| TurboQuant 4-bit | 8× smaller | **0.966 → 1.000** | 99.8 % |
| TurboQuant 2-bit | 16× smaller | **0.902 → 0.992** | 99.8 % |

TurboQuant does what the paper promises: at each bit budget it keeps the cosine neighbours better than
int8 or binary. But the recall of the pairs the product actually scores does not move at all, because the
pairs that matter are near-duplicates that every method still finds within 200 candidates.

Elasticsearch's built-in options give the same picture (local index, top-10 overlap with exact search):
`hnsw` 0.993, `int8_hnsw` 0.986, `int4_hnsw` 0.950 (0.996 with rescoring), `bbq_hnsw` 0.878 (0.995 with
rescoring). Note that Elasticsearch 8.18 already applies `int8_hnsw` by default when a mapping does not
choose: many indexes are compressed without anyone deciding it.

---

## 4. Verdicts

**A. Similar provisions: TurboQuant cannot help.**
* The score of record is edit distance, not cosine. Compressed vectors could at most choose candidates, and
  compression changes recall by less than 0.3 %. The limit is the gap between cosine and edit distance,
  not vector precision.
* The cost is **how many pairs are kept** (70 % at a floor of 30). No vector technique changes that number.
  The real levers are the floor (a product decision) and a candidate filter. Trigram matching, already
  available in PostgreSQL with an index, does as well as the embeddings.

**B. Memo suggestions: measured too (`memo_bench/`). TurboQuant does not help here either.**

The memo path was rebuilt locally: 1,238 real comments (685 questions, 553 responses), the platform's model
(`text-embedding-3-large`, 3,072-d), the same Qdrant version, exact search with the firm filter, own thread
excluded, 50 results, floors 0.5 then 0.42. A known-answer check first: Qdrant's results equal a numpy
exact search on 200 of 200 queries.

*Where the time goes in one suggestion request:*

| Step | Time (p50) |
|---|---|
| Embedding the new comment (API call) | ~240 ms |
| Vector search, 1,238 memos, exact | **~14 ms** |
| LLM relevance check over the top 12 (similar-sized call) | **~3,800 ms** |

The vector search is well under 1 % of the request, and suggestions are computed in the background and cached,
not while the user waits.

*When would the vector search matter?* All memos in one firm (the worst case for an exact search):

| Memos in the firm | Exact search p50 | Vector RAM, float32 | With Qdrant's built-in int8 |
|---|---|---|---|
| 10,000 | 21 ms | 117 MB | 29 MB |
| 50,000 | 53 ms | 586 MB | 146 MB |
| 200,000 | 339 ms | 2.3 GB | 0.6 GB (255 ms) |

Only around 200,000 memo comments in a single firm does the search reach a third of a second, still well below
the LLM step. Even then, Qdrant's built-in options (an HNSW index instead of exact search, int8 quantization)
already solve it with the version in use today.

*Does compression change the suggestions?* Score floor 0.5, pairs from each question's top 50:

| Method | Memory per vector | Top-10 overlap | Suggestions dropped at 0.5 |
|---|---|---|---|
| float32 | 12 KB | 1.000 | 0 |
| scalar int8 (offline) | 3 KB | 0.965 | 1,438 of 22,193 (6.5 %) |
| **TurboQuant 4-bit** | 1.5 KB | **0.983** | **244 (1.1 %)** |
| TurboQuant 2-bit | 0.75 KB | 0.945 | 3,574 (16 %) |
| Qdrant int8 / binary **with rescoring** | — | — | **0** |

TurboQuant again beats plain int8 at the same job, as the paper promises. But Qdrant's own quantization with
rescoring loses nothing at all and needs no new code, so there is no gap left for TurboQuant to fill.

*What the benchmark did find:* random pairs of memos already score 0.38 on average (95th percentile 0.56), so
the 0.5 floor lets through about 152 candidates per question. The quality of the suggestions depends on the
re-ranker and the LLM check, not on vector precision. That is where improvement work would pay.

---

## 5. Checklist: before you compress vectors

1. **Where are the vectors written and read?** If nothing reads them, the question is whether to keep paying
   for them, not how to compress them.
2. **Is the product's score a vector score?** If it is edit distance, BM25 or a business rule, vectors can
   only pre-filter, and a lexical pre-filter may do as well.
3. **What dominates the cost?** Count the stored items, model calls and network round trips. Compression
   shrinks bytes per vector, not the number of anything.
4. **Is search exact or approximate?** Exact search on small filtered sets is rarely memory-bound.
5. **Validate the instruments.** Check every measurement against a known answer before trusting it. Ours
   caught two bugs that would otherwise have produced plausible but wrong tables.

## 6. Run it

See `es_bench/README.md`: `docker compose up -d`, `python canary.py` (must print `ALL PASS`), then
`python run_es_bench.py --es <source> --local-es http://127.0.0.1:9201 --out <folder>`. The source cluster is
only read; indexes are created only on the local cluster.
