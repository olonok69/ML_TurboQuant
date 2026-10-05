# Provision search checks: instructions for the platform repository

This file travels with the benchmark notebook into the platform repository, the one that holds the Qdrant integration, the "similar provisions" feature and the auto-merge job. It tells whoever does the checks (a developer or a coding agent such as Cursor) what to look at in the code, what to read from Qdrant, how to run the notebook, and what to report back.

**The two problems to explain:**

1. **A bottleneck.** Something in provision search or auto-merge is slow. We do not yet know which step.
2. **Poor search quality.** "Similar provisions" returns results that are not similar enough, and auto-merging may join the wrong provisions or miss real duplicates.

The same checklists are in the shared doc *Provision search: bottleneck and quality checks*. This file is the hands-on version for the repository.

---

## 0. Rules

* **Production is read-only.** Against the production Qdrant, only read: `GET /collections/...`, `scroll`, `query_points` / `search`. Never create, update, delete, re-index, change a collection's config or upsert points, even "temporarily".
* **The notebook runs on Google Colab only**, never on a local machine. Locally you only read code, run the grep commands below and edit files.
* **Data policy first.** Running the notebook copies a sample of provision vectors (and provision text if `TEXT_FIELD` is set) into Colab. Get written confirmation from the data protection owner that this is allowed before running it. If it is not, stop at section 2 of this file and report.
* **Secrets never go in files or logs.** The Qdrant URL and API key go into Colab › Secrets as `QDRANT_URL` and `QDRANT_API_KEY`.
* **Report facts, not fixes.** Do not change application code as part of these checks. Write findings into the report (section 5); fixes come afterwards, agreed with the team.

---

## 1. Files to add to the platform repository

Copy these from the TurboQuant repository into a folder such as `tools/provision_checks/`:

| File | Needed for | Notes |
|---|---|---|
| `PROVISION_SEARCH_CHECKS.md` | Everyone | This file |
| `provision_search_benchmark.ipynb` | Running the benchmark in Colab | Self-contained: it embeds the two `.py` files below in `%%writefile` cells |
| `provision_bench.py` | Reading or changing the benchmark logic | Search, metrics, compression methods, auto-merge helpers |
| `turboquant_core.py` | Same | TurboQuant reference implementation used by `provision_bench.py` |
| `provision_search_benchmark.py` | Editing the notebook as a script | Optional. If you edit it, regenerate the notebook (see the TurboQuant repo's `build_notebooks.py`) or edit the notebook directly |

Add `tools/provision_checks/results/` and any `*.npz` vector caches to `.gitignore`: results come back as CSVs in the report, and vectors must never be committed.

---

## 2. Code checks in the platform repository

Run the searches from the repository root, read the code they point to, and fill in the report. Adjust the patterns to the language of the codebase. The examples are for Python; for TypeScript, search `*.ts` and use the JS client names (`search`, `query`, `scroll`, `recommend`).

### 2.1 Find the Qdrant integration

```bash
# client creation, collection names, every call to Qdrant
rg -n "QdrantClient|qdrant_client|@qdrant/js-client|QDRANT_" --glob '!**/node_modules/**'
rg -n "\.(search|query_points|query_batch_points|search_batch|recommend|scroll|retrieve|upsert|upload_collection|update_collection|create_collection)\(" --glob '!**/node_modules/**'
rg -n "collection_name\s*=|collectionName" --glob '!**/node_modules/**'
```

For each call that touches the provisions collection, record in the report: file and line, which feature uses it (similar provisions, auto-merge, ingestion), and the parameters below.

| Check | What to look for | Why it matters |
|---|---|---|
| Client lifetime | Is `QdrantClient` created once (module level, dependency injection) or per request? | A client per request adds connection setup to every search |
| `exact=True` / high `hnsw_ef` | `SearchParams(exact=True)`, `hnsw_ef=` | Exact search ignores the index and scans everything |
| `with_vectors` | `with_vectors=True` on search or scroll in request paths | Returns full vectors over the network for nothing |
| `with_payload` | `with_payload=True` with large payloads (full provision text) | Large responses; ask only for the fields the screen needs |
| `limit` | Large limits trimmed later in code | Fetches far more than shown |
| Loops of single calls | A `for` loop calling `search` / `query_points` once per item | Use `query_batch_points` (batch) instead |
| Filters | `query_filter=` / `filter:` fields used | Every filtered field needs a payload index (checked in section 3) |
| Quantization params | `QuantizationSearchParams(rescore=..., oversampling=..., ignore=...)` | If the collection is quantized and rescore is off, ranking uses only compressed vectors |
| `score_threshold` | Where it is set and to what | Thresholds silently cut results; record the values |

### 2.2 Find the embedding step

```bash
rg -n "embed|Embedding|encode\(|SentenceTransformer|text-embedding|openai\.embeddings|bedrock|vertex|cohere|voyage" --glob '!**/node_modules/**'
rg -n "max_length|max_seq_length|truncat|chunk_size|chunk_overlap|split_text|TextSplitter" --glob '!**/node_modules/**'
```

Record:

* The embedding model name and version, and where it is configured. Confirm the **same model** embeds stored provisions and queries. Check whether the model was ever changed without re-embedding the whole collection (git history of the config: `git log -p -- <config file>`).
* The model's input limit and how long provisions are handled: truncated, chunked (chunk size, overlap, how chunk scores are combined) or rejected.
* Exactly **what text** is embedded per provision: full text, title, text plus headings, numbering or boilerplate. Paste the function that builds the string.
* Whether vectors are normalized before upsert, and the collection distance (section 3). Dot product on unnormalized vectors ranks long provisions first.
* Whether edited provisions are re-embedded and upserted, or keep stale vectors.
* For query-time embedding in "similar provisions": add (or find) a timing log around the embedding call. It is often slower than the vector search itself.

### 2.3 Find the "similar provisions" flow

```bash
rg -n -i "similar.?provision|similarProvision|similar_provisions|more.?like.?this" --glob '!**/node_modules/**'
```

Trace one request from the API endpoint to the response and record every step with its approximate cost: database lookups, embedding, Qdrant call(s), post-processing (re-ranking, de-duplication, filtering in code), payload or text loading from another store. If the code searches by an existing provision, check whether it re-embeds the provision text instead of using the vector already stored in Qdrant (Qdrant can search by point id).

### 2.4 Find the auto-merge job

```bash
rg -n -i "auto.?merge|automerge|merge_provision|dedup|duplicate|pairwise|similarity_threshold|merge_threshold" --glob '!**/node_modules/**'
rg -n -i "rapidfuzz|fuzzywuzzy|difflib|SequenceMatcher|levenshtein|jaro|jaccard|tfidf|TfidfVectorizer|cosine_similarity|pdist|cdist" --glob '!**/node_modules/**'
```

This is the most likely bottleneck. Record:

| Check | What to look for |
|---|---|
| How candidates are generated | Nested loops over all provisions, `itertools.combinations`, `cdist` / `cosine_similarity` on the full matrix → every pair is compared and the cost grows with the square of the count. A Qdrant top-k query per provision (with `score_threshold`) compares only near neighbours. |
| What "pairwise textual similarity" is | Embedding cosine from Qdrant vectors, or a **string** comparison (fuzzy ratio, edit distance, TF-IDF, Jaccard). String comparison over many pairs is slow and ignores the embeddings already stored. |
| Re-embedding | Does the job call the embedding model for text that already has a vector in Qdrant? |
| Incremental runs | Does it process only new or changed provisions, or the whole collection each run? |
| Threshold | The value, where it is configured, and any comment, ticket or commit explaining how it was chosen (`git log -S "<threshold value>"`) |
| Transitivity | If A merges with B and B with C, is A merged with C? Look for union-find, connected components or graph code. Chains merge provisions that differ. |
| Audit and undo | Are merges logged with their scores, and can they be reverted? |
| Run time | Find job logs or scheduler history with run durations and the number of provisions processed |

Write down the `MERGE_THRESHOLD` value: the notebook needs it.

---

## 3. Qdrant checks (read-only)

From the Qdrant dashboard or with read-only calls (`curl -H "api-key: $QDRANT_API_KEY" $QDRANT_URL/collections/<name>`). The notebook's section 1 prints the same information with automatic flags.

| Check | Healthy | Problem |
|---|---|---|
| Qdrant version | Known; 1.18 or later is needed for TurboQuant | Very old versions lack batch query, rescoring options |
| `status` | green | yellow or grey: the optimizer is still working |
| `indexed_vectors_count` vs `points_count` | close | a large gap: part of the data is searched without the index |
| Vectors / HNSW `on_disk` | in RAM, or quantized copy in RAM | on disk with no quantized copy: disk reads on every search |
| Node RAM vs vectors + index size | vectors fit with room to spare | swapping or page faults |
| `hnsw_config` (`m`, `ef_construct`) | defaults (16, 100) or tuned | very low values lose recall |
| `quantization_config` | none, or set with rescoring used in code | set, but code never rescores |
| Segments | a handful of large segments | dozens of tiny segments |
| Payload indexes | one per filtered field | filtered fields without an index |
| Collection distance | matches the embedding model (usually Cosine) | Dot with unnormalized vectors; Euclid with a cosine model |

Also record p50 and p95 search latency from Qdrant's `/metrics` endpoint or the cloud dashboard, if available.

---

## 4. Run the benchmark notebook (Google Colab)

Only after the data-policy confirmation in section 0.

1. Open colab.research.google.com › File › Upload notebook › `provision_search_benchmark.ipynb`.
2. Add Colab Secrets (key icon on the left): `QDRANT_URL`, `QDRANT_API_KEY`. Enable notebook access for both.
3. In the **Settings** cell, set:
   * `COLLECTION`: the provisions collection name (section 2.1).
   * `VECTOR_NAME`: the vector name if the collection uses named vectors, else `None`.
   * `MERGE_THRESHOLD`: the value from the auto-merge code (section 2.4).
   * `TEXT_FIELD`: the payload field with provision text, **only if** the data owner allowed text in Colab; else leave `None`.
   * `N_SAMPLE`: keep 100,000 unless the collection is smaller (it will take all of it).
4. Optional: upload expert-labelled CSVs through the Files pane (both use Qdrant point ids):
   * `similar_labels.csv` with columns `query_id,relevant_id`: for 50 to 100 provisions, the provisions that should come back as similar.
   * `merge_labels.csv` with columns `id_a,id_b,same`: 200 or more pairs, `same` = 1 for true duplicates, 0 otherwise, including near misses.
5. Runtime › Run all. A CPU runtime is enough for 100,000 vectors; a GPU runtime makes sections 3 to 5 faster.
6. The last cell downloads `provision_benchmark_results.zip` (CSV numbers only). Put it in `tools/provision_checks/results/` locally, unzipped, and do not commit it unless the team agrees.

What each notebook section answers:

| Section | Question |
|---|---|
| 1. The live collection | Is the collection configured for fast search? (flags) |
| 2. Load a sample | (pulls vectors with `scroll`, read-only) |
| 3. Embedding health | Are the embeddings themselves healthy: normalization, duplicates, hub provisions, neighbours vs random pairs, text spot check? |
| 4. Compression benchmark | Recall, memory and speed of float32, scalar int8, binary and TurboQuant 4-bit / 2-bit, with and without rescoring |
| 5. Auto-merge stability | How many merge decisions flip at our threshold under each option; how many pairs sit within ±0.01 of it |
| 6. Live search | Production latency (p50, p95) and how much of the exact top 10 production returns, with quantization on, ignored and exact |
| 7. Real Qdrant test | The same options in a throwaway Qdrant 1.18 started inside Colab (never production) |
| 8. Labelled evaluation | Recall against expert labels; precision/recall curve of the merge threshold |
| 9. Results | CSVs zipped for download |

If section 7 cannot download Qdrant, set `QDRANT_VERSION` to an existing release from github.com/qdrant/qdrant/releases (1.18 or later) or set `RUN_LOCAL_QDRANT = False`. If section 1 fails on a client attribute, note the `qdrant-client` version (`pip show qdrant-client`) in the report.

### How to read the results

| Result | Meaning | Direction |
|---|---|---|
| Live `default` overlap with exact well below 1.0, `ignore quantization` close to 1.0 | Quantization loses results | Search with `rescore=True`, `oversampling` 2 to 4 |
| Both live overlaps low | The HNSW index loses results | Raise `hnsw_ef` at search time, or `m` / `ef_construct` |
| Live p95 high but overlap fine | Infrastructure: disk reads, RAM, segments, network | Section 3 flags; quantization with a copy in RAM; more RAM |
| Embedding health: random pairs score almost as high as neighbours, high `mean_direction_norm`, many hubs | Embeddings do not separate provisions well (boilerplate, model fit) | Change what text is embedded, then the model; consider hybrid search (dense + keyword) and a cross-encoder re-ranker |
| Labelled recall low even for `float32 exact` | The embeddings, not the index or compression, are the quality problem | As above |
| Many pairs within ±0.01 of the merge threshold, or flips under compression | Auto-merge decisions are fragile | Calibrate the threshold on `merge_labels.csv`; always use exact (rescored) scores for merge decisions |
| TurboQuant 4-bit with rescoring ≈ float32 recall at 8x less memory | Compression is safe on our data | Candidate for production if memory or disk reads are the bottleneck; re-index required |

---

## 5. Report back

Create `tools/provision_checks/REPORT.md` with these headings and send it (plus the results CSVs) to the team. Keep it factual; quote file paths and line numbers.

```markdown
# Provision search checks: report

Date:            Repository commit:          Done by:

## Environment
- Qdrant version / deployment / node RAM:
- Collection: name, points, vector size, distance, quantization, HNSW, on_disk, segments, payload indexes:
- Embedding model + version, input limit, normalization:
- qdrant-client version:

## Bottleneck
- Slowest step (with timings, p50/p95):
- Similar provisions request path (steps and costs):
- Auto-merge: candidate generation, similarity used, run time vs number of provisions:
- Code findings (file:line, issue):

## Quality
- Text embedded per provision and chunking:
- Embedding health numbers (section 3 of the notebook):
- Live overlap with exact search (section 6):
- Labelled recall, if labels were provided (section 8):
- Spot-check verdict from a domain expert:

## Auto-merge
- Threshold value and origin:
- Pairs near the threshold, flips under compression (section 5):
- Precision/recall at the current threshold, if labels were provided:
- Transitivity / chaining behaviour:

## Compression options (section 4 and 7)
- Table: method, rescoring, recall@10, memory, latency:

## Recommended next steps (ranked)
1.
2.
3.

## Open questions
-
```

Do not include provision text, vectors, URLs or keys in the report.
