# %% [markdown]
# # Provision search: compression, quality and auto-merge benchmark (Google Colab)
#
# This notebook benchmarks **our own provision embeddings** from Qdrant. It answers three questions:
#
# 1. **Bottleneck.** How is the live collection configured, and how fast and how accurate is the live search compared with an exact search?
# 2. **Quality.** Do the embeddings themselves look healthy (normalization, duplicates, boilerplate "hubs", how separated neighbours are from random pairs)? With an optional labelled test set, how good are the results?
# 3. **Compression and auto-merge.** What would scalar, binary or TurboQuant quantization do to recall, memory and speed on our data, and how many auto-merge decisions would flip at our threshold?
#
# It is the companion of the doc *Provision search: bottleneck and quality checks*. It is separate from the TurboQuant workshop demos.
#
# **Safety and data.**
# * Everything against the production Qdrant is **read-only**: `get_collection`, `scroll` and `query_points`. Nothing is created, changed or deleted there.
# * Section 7 creates collections only in a **temporary Qdrant server started inside this Colab runtime**, which disappears with the runtime.
# * The sample of provision vectors (and, if `TEXT_FIELD` is set, some provision text) is downloaded into this Colab runtime. **Check this is allowed by our data policy before running.** Do not share an executed copy outside the team if it shows provision text.
#
# **Credentials:** add `QDRANT_URL` and `QDRANT_API_KEY` in Colab › Secrets (key icon on the left). They are never printed.
#
# **Runtime:** CPU is enough for up to about 100k vectors. A GPU runtime makes sections 3 to 5 faster.

# %%
import subprocess, sys, importlib
for mod, spec in [("qdrant_client", "qdrant-client"), ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("requests", "requests")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", spec], check=True)
try:  # optional: TurboQuant with SIMD kernels
    import turbovec  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "turbovec"], check=False)

# %% [markdown]
# ## Helper modules
# The next two cells write `turboquant_core.py` (the TurboQuant reference implementation) and `provision_bench.py` (search, metrics and auto-merge helpers used below).

# %%
# Needs turboquant_core.py in the same folder (in the notebook this cell writes it).

# %%
# Needs provision_bench.py in the same folder (in the notebook this cell writes it).

# %% [markdown]
# ## Settings
# Edit this cell, then Runtime › Run all.

# %%
import os
SOURCE = "qdrant"            # "qdrant" = read a sample from the live collection; "npy" = load NPY_VECTORS / NPY_IDS instead
COLLECTION = "provisions"     # name of the provisions collection (check it in the Qdrant dashboard)
VECTOR_NAME = None            # set to the vector name if the collection uses named vectors, e.g. "dense"
TEXT_FIELD = None             # payload field with the provision text, e.g. "text"; None = no text is downloaded or shown
N_SAMPLE = 100_000            # vectors to pull for the offline benchmark (more = slower, closer to production)
N_QUERIES = 1_000             # provisions used as queries ("find provisions similar to this one")
K = 10                        # results shown by "similar provisions"
MERGE_THRESHOLD = 0.90        # the similarity threshold the auto-merge job uses (put the real value here)
N_LIVE_QUERIES = 100          # queries sent to the live collection in section 6 (read-only)
RUN_LOCAL_QDRANT = True       # section 7: start a temporary Qdrant 1.18 inside Colab and test its quantization options
QDRANT_VERSION = "v1.18.0"    # release used in section 7 (needs 1.18 or later for TurboQuant)
NPY_VECTORS, NPY_IDS = "provision_vectors.npy", "provision_ids.npy"   # only for SOURCE = "npy"
SIMILAR_LABELS_CSV = "similar_labels.csv"   # optional, section 8: columns query_id, relevant_id
MERGE_LABELS_CSV = "merge_labels.csv"       # optional, section 8: columns id_a, id_b, same (1 = same provision)


def secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return os.environ.get(name)


QDRANT_URL, QDRANT_API_KEY = secret("QDRANT_URL"), secret("QDRANT_API_KEY")
print("Qdrant URL set:", bool(QDRANT_URL), "| API key set:", bool(QDRANT_API_KEY))

# %%
import math, time, json
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print
import turboquant_core, provision_bench as pb
importlib.reload(turboquant_core); importlib.reload(pb)
pd.set_option("display.max_columns", 30); pd.set_option("display.width", 200)
print("device:", pb.DEVICE)
RESULTS = {}   # tables saved at the end

# %% [markdown]
# ## 1. The live collection (read-only)
# Configuration that matters for speed and quality, with automatic flags for the checks in the doc's "Bottleneck checks" section.

# %%
client = None
DIST = "Cosine"
if SOURCE == "qdrant":
    from qdrant_client import QdrantClient, models
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    info = client.get_collection(COLLECTION)
    cfg = info.config
    vparams = cfg.params.vectors
    if isinstance(vparams, dict):  # named vectors
        assert VECTOR_NAME in vparams, f"named vectors {list(vparams)}: set VECTOR_NAME to one of them"
        vparams = vparams[VECTOR_NAME]
    DIST = str(vparams.distance.value if hasattr(vparams.distance, "value") else vparams.distance)
    hnsw = vparams.hnsw_config or cfg.hnsw_config
    quant = vparams.quantization_config or cfg.quantization_config
    live = dict(status=str(info.status), points=info.points_count, indexed_vectors=info.indexed_vectors_count,
                segments=info.segments_count, dim=vparams.size, distance=DIST,
                vectors_on_disk=getattr(vparams, "on_disk", None), hnsw_m=hnsw.m, hnsw_ef_construct=hnsw.ef_construct,
                hnsw_on_disk=hnsw.on_disk, quantization=json.dumps(quant.model_dump(exclude_none=True)) if quant else None,
                payload_indexes=", ".join(f"{k} ({v.data_type})" for k, v in (info.payload_schema or {}).items()) or None)
    display(pd.DataFrame([live]).T.rename(columns={0: "value"}))
    flags = []
    if "green" not in live["status"].lower():
        flags.append("Status is not green: the optimizer is still working, searches may be slower.")
    if live["points"] and live["indexed_vectors"] is not None and live["indexed_vectors"] < 0.9 * live["points"]:
        flags.append(f"Only {live['indexed_vectors']:,} of {live['points']:,} vectors are in the HNSW index: part of the data is searched without it.")
    if live["vectors_on_disk"] and not quant:
        flags.append("Vectors are on disk with no quantized copy in RAM: every search reads from disk.")
    if live["segments"] and live["points"] and live["segments"] > 32 and live["points"] / live["segments"] < 20_000:
        flags.append(f"{live['segments']} segments for {live['points']:,} points: many small segments slow every search.")
    if not live["payload_indexes"]:
        flags.append("No payload indexes: any filtered search scans payloads.")
    if quant:
        flags.append("Quantization is on: make sure the code searches with rescore enabled (section 6 measures the effect).")
    print("\n".join("FLAG: " + f for f in flags) or "No configuration flags.")
    RESULTS["live_collection"] = pd.DataFrame([live])
assert DIST.lower().startswith(("cos", "dot")), "This notebook handles Cosine and Dot collections; for Euclid, ask before using it."

# %% [markdown]
# ## 2. Load a sample of provision vectors
# Pulled with `scroll` (read-only) in id order and cached in this runtime. Note that the first `N_SAMPLE` points by id may not be a random sample of the collection.

# %%
TEXTS = None
if SOURCE == "qdrant":
    cache = f"/content/{COLLECTION}_{N_SAMPLE}.npz" if os.path.isdir("/content") else f"{COLLECTION}_{N_SAMPLE}.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); X_raw, IDS = z["X"], z["ids"]; TEXTS = z["texts"] if "texts" in z else None
    else:
        vecs, ids, texts, offset, t0 = [], [], [], None, time.time()
        while len(ids) < N_SAMPLE:
            pts, offset = client.scroll(COLLECTION, limit=min(1000, N_SAMPLE - len(ids)), offset=offset,
                                        with_vectors=[VECTOR_NAME] if VECTOR_NAME else True,
                                        with_payload=[TEXT_FIELD] if TEXT_FIELD else False)
            for p in pts:
                v = p.vector[VECTOR_NAME] if VECTOR_NAME else p.vector
                vecs.append(v); ids.append(p.id)
                if TEXT_FIELD:
                    texts.append(str((p.payload or {}).get(TEXT_FIELD, ""))[:2000])
            if offset is None:
                break
        X_raw, IDS = np.asarray(vecs, dtype=np.float32), np.asarray(ids, dtype=object)
        TEXTS = np.asarray(texts, dtype=object) if TEXT_FIELD else None
        np.savez(cache, X=X_raw, ids=IDS, **({"texts": TEXTS} if TEXTS is not None else {}))
        print(f"scrolled {len(IDS):,} points in {time.time() - t0:.0f}s")
else:
    X_raw, IDS = np.load(NPY_VECTORS).astype(np.float32), np.load(NPY_IDS, allow_pickle=True)
X = pb.normalize(X_raw) if DIST.lower().startswith("cos") else X_raw.astype(np.float32)
N, D = X.shape
QIDX = np.random.default_rng(0).choice(N, min(N_QUERIES, N), replace=False).astype(np.int64)
print(f"{N:,} vectors × {D} dims ({X.nbytes / 2**20:.0f} MB as float32), {len(QIDX):,} queries, distance {DIST}")

# %% [markdown]
# ## 3. Exact neighbours and embedding health
# Exact top-50 neighbours of every query (its own row excluded) are the ground truth for everything below. The health numbers need no labels:
#
# | Number | What a problem looks like |
# |---|---|
# | `already_normalized` | `False` with a Dot collection: long vectors win regardless of meaning |
# | `random_pair_cos_mean` | Close to the neighbours' scores: everything looks similar to everything (boilerplate, a shared direction) |
# | `mean_direction_norm` | Above ~0.3: vectors share one dominant direction, which compresses all scores together |
# | `exact_duplicate_vectors`, `near_duplicate_queries_pct` | Many: duplicates crowd the top results; they are also the auto-merge candidates |
# | `top1pct_hubs_share_of_top10_slots` | Well above 0.01: a few "hub" provisions appear in many unrelated result lists |
# | `effective_dim_90pct` | Very low compared with the dimension: the model uses little of its space for our texts |

# %%
t = time.time(); S_TRUE, I_TRUE = pb.exact_topk(X, QIDX, 50); print(f"exact search: {time.time() - t:.1f}s")
health, rand_cos, occ = pb.embedding_health(X_raw, X, S_TRUE, I_TRUE)
display(pd.DataFrame([health]).T.rename(columns={0: "value"}))
RESULTS["embedding_health"] = pd.DataFrame([health])

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ax[0].hist(rand_cos, 80, alpha=.6, density=True, label="random pairs")
ax[0].hist(S_TRUE[:, 0], 80, alpha=.6, density=True, label="nearest neighbour")
ax[0].hist(S_TRUE[:, K - 1], 80, alpha=.6, density=True, label=f"{K}th neighbour")
ax[0].axvline(MERGE_THRESHOLD, c="k", ls="--", lw=1, label="merge threshold")
ax[0].set(title="Similarity scores: neighbours vs random pairs", xlabel="score"); ax[0].legend(fontsize=8)
ax[1].hist(np.linalg.norm(X_raw, axis=1), 80); ax[1].set(title="Vector norms (before normalization)", xlabel="norm")
ax[2].plot(np.sort(occ)[::-1][:200]); ax[2].set(title=f"Hubness: times each provision appears in a top-10 (top 200)", xlabel="provision rank", ylabel="appearances")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Spot check for a domain expert** (only when `TEXT_FIELD` is set): five random provisions with their top-5 neighbours, and the biggest hubs. Judge whether the neighbours are really similar.

# %%
if TEXTS is not None:
    short = lambda s: (s[:220] + "…") if len(s) > 220 else s
    for j in np.random.default_rng(1).choice(len(QIDX), 5, replace=False):
        q = QIDX[j]
        print(f"\n=== QUERY {IDS[q]}: {short(TEXTS[q])}")
        for s, i in zip(S_TRUE[j, :5], I_TRUE[j, :5]):
            print(f"  {s:.3f}  {IDS[i]}: {short(TEXTS[i])}")
    print("\n=== BIGGEST HUBS (appear in many unrelated top-10 lists)")
    for i in np.argsort(-occ)[:5]:
        print(f"  {occ[i]:4d}×  {IDS[i]}: {short(TEXTS[i])}")
else:
    print("TEXT_FIELD not set: no provision text downloaded.")

# %% [markdown]
# ## 4. Compression benchmark (offline, on the sample)
# Each method mimics a Qdrant quantization option: the query stays in float32, stored vectors are compressed. "Rescore" re-ranks an oversampled candidate list with the exact vectors, like `rescore=True` with `oversampling` in Qdrant.
#
# This is an **indicative simulation**: Qdrant's own TurboQuant adds extensions (for example a length renormalization, approximated here by the norm correction), and HNSW adds its own approximation. Section 7 measures the real thing.

# %%
methods = [pb.Float32(), pb.ScalarInt8(0.99), pb.Binary1(), pb.TurboQuantRef(4), pb.TurboQuantRef(2), pb.TurboQuantRef(2, renorm=False)]
if D % 8 != 0:
    methods = [m for m in methods if not isinstance(m, pb.TurboQuantRef)]
    print("dimension not divisible by 8: TurboQuant reference skipped")
try:
    import turbovec  # noqa: F401
    methods += [pb.Turbovec(4), pb.Turbovec(2)]
except ImportError:
    print("turbovec not available: skipped")
rows = []
for m in methods:
    try:
        rows += pb.run_method(m, X, QIDX, I_TRUE[:, :K], k=K, oversampling=(1, 2, 4))
    except Exception as e:
        print(f"{m.name}: failed ({e})")
bench = pd.DataFrame(rows)
display(bench)
RESULTS["compression_benchmark"] = bench

fig, ax = plt.subplots(figsize=(9, 5))
for name, g in bench.groupby("method"):
    ax.plot(g.compression, g[f"recall@{K}"], "o", label=name)
    for _, r in g.iterrows():
        ax.annotate(r.search.replace("rescore, oversampling ", "rs ").replace("no rescore", "raw"), (r.compression, r[f"recall@{K}"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set(xscale="log", xlabel="compression vs float32 (log)", ylabel=f"recall@{K} vs exact", title="Recall vs memory on our provision vectors")
ax.legend(fontsize=7); ax.grid(alpha=.3); plt.show()

# %% [markdown]
# ## 5. Auto-merge stability
# Candidate pairs = each query's exact neighbours scoring at least `MERGE_THRESHOLD − 0.1`. For each compression method we score the same pairs the way a quantized search would (float query against the compressed stored vector) and count decisions that flip at the threshold. With `rescore=True` in Qdrant the final scores are exact, so these flips disappear.
#
# The threshold sweep shows how many pairs would merge at each threshold, and how many pairs sit within ±0.01 of the current one (decisions that any small change can flip).

# %%
A, B, S_PAIR = pb.candidate_pairs(QIDX, S_TRUE, I_TRUE, MERGE_THRESHOLD - 0.1)
near = int((np.abs(S_PAIR - MERGE_THRESHOLD) <= 0.01).sum())
print(f"{len(S_PAIR):,} candidate pairs, {int((S_PAIR >= MERGE_THRESHOLD).sum()):,} at or above the threshold, {near:,} within ±0.01 of it")
flip_rows = []
for m in [pb.ScalarInt8(0.99), pb.TurboQuantRef(4), pb.TurboQuantRef(2), pb.TurboQuantRef(2, renorm=False)]:
    if isinstance(m, pb.TurboQuantRef) and D % 8 != 0:
        continue
    m.build(X)
    flip_rows.append(dict(method=m.name, **pb.merge_flips(S_PAIR, pb.approx_pair_scores(m, X, A, B), MERGE_THRESHOLD)))
flips = pd.DataFrame(flip_rows); display(flips); RESULTS["merge_flips"] = flips

ths = np.round(np.arange(MERGE_THRESHOLD - 0.1, min(MERGE_THRESHOLD + 0.1, 1.0) + 1e-9, 0.005), 3)
sweep = pd.DataFrame(dict(threshold=ths, pairs_that_would_merge=[int((S_PAIR >= t).sum()) for t in ths]))
fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(sweep.threshold, sweep.pairs_that_would_merge, "o-")
ax.axvline(MERGE_THRESHOLD, c="k", ls="--", lw=1); ax.set(xlabel="threshold", ylabel="pairs that would merge", title="Merges vs threshold (exact scores, sampled queries)")
plt.show(); RESULTS["merge_threshold_sweep"] = sweep

# %% [markdown]
# ## 6. Live search: latency and accuracy (read-only)
# The same queries against the production collection in up to three modes: the default search, the same search ignoring quantization (only when the collection is quantized), and an exact search. The overlap with the exact results shows how much the index and the quantization lose; the latencies show what the Qdrant call costs on its own. Embedding the query text is **not** included: time it in the application.

# %%
if SOURCE == "qdrant":
    lq = QIDX[:N_LIVE_QUERIES]
    modes = {"default": None}
    if quant:
        modes["ignore quantization"] = models.SearchParams(quantization=models.QuantizationSearchParams(ignore=True))
        modes["rescore, oversampling 2x"] = models.SearchParams(quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0))
    modes["exact"] = models.SearchParams(exact=True)
    live_ids, live_ms = {}, {}
    for mode, sp in modes.items():
        res, ms = [], []
        for q in lq:
            t = time.perf_counter()
            r = client.query_points(COLLECTION, query=X_raw[q].tolist(), using=VECTOR_NAME, limit=K + 1,
                                    search_params=sp, with_payload=False, with_vectors=False)
            ms.append((time.perf_counter() - t) * 1000)
            res.append([p.id for p in r.points if p.id != IDS[q]][:K])
        live_ids[mode], live_ms[mode] = res, ms
    ex = live_ids["exact"]
    live_rows = [dict(mode=m, p50_ms=round(float(np.percentile(live_ms[m], 50)), 1), p95_ms=round(float(np.percentile(live_ms[m], 95)), 1),
                      **{f"overlap@{K}_with_exact": round(float(np.mean([len(set(a) & set(b)) / K for a, b in zip(live_ids[m], ex)])), 4)})
                 for m in modes]
    live_df = pd.DataFrame(live_rows); display(live_df); RESULTS["live_search"] = live_df
    print("Reading it: overlap well below 1.0 in 'default' but close to 1.0 in 'ignore quantization' = quantization loses results "
          "(turn on rescore); low in both = the HNSW index loses results (raise hnsw_ef or m). Latency includes network round trips.")

# %% [markdown]
# ## 7. Optional: real Qdrant quantization, in a temporary local server
# Starts Qdrant `QDRANT_VERSION` inside this Colab runtime (nothing touches production), loads the sample into one collection per quantization option, and measures recall against the exact neighbours with and without rescoring. TurboQuant needs Qdrant 1.18 or later. If a configuration is not accepted by that version, it is reported and skipped.

# %%
if RUN_LOCAL_QDRANT:
    import requests, tarfile, urllib.request
    from qdrant_client import QdrantClient, models
    LOCAL = "http://localhost:6333"
    exe = "/tmp/qdrant/qdrant"
    if not os.path.exists(exe):
        url = f"https://github.com/qdrant/qdrant/releases/download/{QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz"
        os.makedirs("/tmp/qdrant", exist_ok=True)
        try:
            urllib.request.urlretrieve(url, "/tmp/qdrant/q.tgz"); tarfile.open("/tmp/qdrant/q.tgz").extractall("/tmp/qdrant")
        except Exception as e:
            print(f"Could not download {url} ({e}). Pick an existing release at github.com/qdrant/qdrant/releases and set QDRANT_VERSION.")
    if os.path.exists(exe):
        env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH="/tmp/qdrant/storage", QDRANT__TELEMETRY_DISABLED="true")
        if "qproc" not in globals() or qproc.poll() is not None:
            qproc = subprocess.Popen([exe], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(60):
                try:
                    requests.get(LOCAL + "/readyz", timeout=1); break
                except Exception:
                    time.sleep(1)
        print("local Qdrant:", requests.get(LOCAL).json().get("version"))
        lc = QdrantClient(url=LOCAL, timeout=300)
        dist = "Cosine" if DIST.lower().startswith("cos") else "Dot"
        configs = {"float32": None,
                   "scalar int8": {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}},
                   "binary 1-bit": {"binary": {"always_ram": True}},
                   "turbo 4-bit": {"turbo": {"bits": "bits4"}},
                   "turbo 2-bit": {"turbo": {"bits": "bits2"}}}
        local_rows = []
        for label, qc in configs.items():
            name = "bench_" + label.replace(" ", "_").replace("-", "")
            requests.delete(f"{LOCAL}/collections/{name}")
            body = {"vectors": {"size": D, "distance": dist}}
            if qc:
                body["quantization_config"] = qc
            r = requests.put(f"{LOCAL}/collections/{name}", json=body)
            if not r.ok:
                print(f"{label}: not accepted by this Qdrant version ({r.status_code}: {r.text[:200]})"); continue
            t = time.time()
            lc.upload_collection(name, vectors=X_raw, ids=list(range(N)), batch_size=512, parallel=2, wait=True)
            for _ in range(600):
                ci = lc.get_collection(name)
                if str(ci.status).lower().endswith("green") and (ci.indexed_vectors_count or 0) >= 0.99 * N:
                    break
                time.sleep(2)
            build_s = time.time() - t
            for rescore in ([False, True] if qc else [False]):
                sp = models.SearchParams(quantization=models.QuantizationSearchParams(rescore=rescore, oversampling=2.0 if rescore else 1.0)) if qc else None
                got, ms = [], []
                for q in QIDX[:300]:
                    t = time.perf_counter()
                    res = lc.query_points(name, query=X_raw[q].tolist(), limit=K + 1, search_params=sp, with_payload=False)
                    ms.append((time.perf_counter() - t) * 1000)
                    got.append([p.id for p in res.points if p.id != q][:K])
                got = np.array([g + [-1] * (K - len(g)) for g in got])
                local_rows.append(dict(config=label, rescore=rescore, build_and_index_s=round(build_s, 1),
                                       p50_ms=round(float(np.percentile(ms, 50)), 2),
                                       **{f"recall@{K}": round(pb.recall_at_k(got, I_TRUE[:300, :K], K), 4),
                                          "top1_found_in_top1": round(pb.r1_at_k(got, I_TRUE[:300, :K], 1), 4)}))
        local_df = pd.DataFrame(local_rows); display(local_df); RESULTS["local_qdrant"] = local_df
        print("Memory per vector, float32:", 4 * D, "bytes; scalar int8 ≈", D, "; turbo 4-bit ≈", D // 2, "; turbo 2-bit ≈", D // 4, "; binary ≈", D // 8)

# %% [markdown]
# ## 8. Optional: labelled evaluation
# Upload CSVs prepared by domain experts (Files pane on the left), using **Qdrant point ids**:
# * `similar_labels.csv` with columns `query_id, relevant_id`: provisions that should come back as similar.
# * `merge_labels.csv` with columns `id_a, id_b, same`: pairs marked 1 (same provision, should merge) or 0 (different).
#
# Labelled ids must be inside the loaded sample.

# %%
row_of = {str(i): r for r, i in enumerate(IDS)}
if os.path.exists(SIMILAR_LABELS_CSV):
    lab = pd.read_csv(SIMILAR_LABELS_CSV, dtype=str)
    rel = {}
    for qa, rb in zip(lab.query_id, lab.relevant_id):
        if qa in row_of and rb in row_of:
            rel.setdefault(row_of[qa], set()).add(row_of[rb])
    lq_rows = np.array(sorted(rel), dtype=np.int64)
    _, I_lab = pb.exact_topk(X, lq_rows, K)
    lab_rows = [dict(method="float32 exact", **{f"label_recall@{K}": round(pb.recall_from_labels(I_lab, lq_rows, rel, K), 4)})]
    for m in [pb.ScalarInt8(0.99), pb.TurboQuantRef(4), pb.TurboQuantRef(2)]:
        m.build(X); _, Im = m.search(X, lq_rows, K)
        lab_rows.append(dict(method=m.name, **{f"label_recall@{K}": round(pb.recall_from_labels(Im, lq_rows, rel, K), 4)}))
    lab_df = pd.DataFrame(lab_rows); display(lab_df); RESULTS["labelled_recall"] = lab_df
    print(f"{len(rel)} labelled queries. Low label recall even for 'float32 exact' = the embeddings, not the index or compression, are the problem.")
else:
    print(f"{SIMILAR_LABELS_CSV} not found: skipped")
if os.path.exists(MERGE_LABELS_CSV):
    ml = pd.read_csv(MERGE_LABELS_CSV, dtype={"id_a": str, "id_b": str})
    ml = ml[ml.id_a.isin(row_of) & ml.id_b.isin(row_of)]
    a = np.array([row_of[i] for i in ml.id_a]); b = np.array([row_of[i] for i in ml.id_b])
    sc = np.einsum("nd,nd->n", X[a], X[b])
    pr = pd.DataFrame(pb.merge_pr_curve(sc, ml.same.astype(int).values, np.round(np.arange(0.70, 1.0, 0.01), 2)))
    display(pr); RESULTS["merge_precision_recall"] = pr
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(pr.threshold, pr.precision, "o-", label="precision"); ax.plot(pr.threshold, pr.recall, "o-", label="recall")
    ax.axvline(MERGE_THRESHOLD, c="k", ls="--", lw=1); ax.set(xlabel="threshold", title=f"Auto-merge on {len(ml)} labelled pairs"); ax.legend(); plt.show()
else:
    print(f"{MERGE_LABELS_CSV} not found: skipped")

# %% [markdown]
# ## 9. Save the results
# Writes one CSV per table to `results/` (numbers only: no vectors, no provision text) and zips them for download. Bring them back to the team, not the executed notebook if it shows provision text.

# %%
os.makedirs("results", exist_ok=True)
for name, df in RESULTS.items():
    df.to_csv(f"results/provision_{name}.csv", index=False)
import shutil
shutil.make_archive("provision_benchmark_results", "zip", "results")
print("saved:", sorted(os.listdir("results")))
try:
    from google.colab import files
    files.download("provision_benchmark_results.zip")
except Exception:
    pass
