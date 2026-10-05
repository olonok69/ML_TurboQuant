# %% [markdown]
# # TurboQuant for vector search: recall, memory and speed (Google Colab)
#
# The TurboQuant paper (arXiv:2504.19874, Section 4.4) shows that a **data-oblivious** quantizer (random rotation plus a fixed Lloyd-Max codebook) beats trained Product Quantization on recall, while making indexing time "virtually zero".
# This notebook reproduces that comparison:
#
# | method | training | notes |
# |---|---|---|
# | Exact float32 (FAISS `IndexFlatIP`) | none | ground truth |
# | **TurboQuant (reference PyTorch)** | none | this repo's `turboquant_core.py`, MSE and inner-product variants |
# | **turbovec** (`pip install turbovec`) | none | Rust + SIMD implementation of TurboQuant |
# | FAISS PQ, 256 centroids per sub-space | k-means | the paper's PQ baseline (LUT256) |
# | FAISS PQ FastScan, 16 centroids | k-means | fastest FAISS PQ, the baseline turbovec benchmarks against |
# | FAISS RaBitQ | light | the paper's other baseline |
# | FAISS SQ4 | min/max | plain 4-bit scalar quantization |
#
# **Data:** the paper's DBpedia entities embedded with OpenAI `text-embedding-3-large` (1536-d), 100k database vectors and 1k queries, streamed from the Hugging Face Hub. Set `DATASET` to `"dbpedia-3072"` for the 3072-d variant or `"20newsgroups-lsa"` for an offline-friendly fallback.
#
# **Runtime:** CPU is fine (turbovec and FAISS are CPU libraries). A GPU only speeds up the reference PyTorch scorer.
#
# **Companion reading:** the workshop guide (`docs/guide/TurboQuant_Technical_Guide_EN.md` in the repository; Spanish version `TurboQuant_Guia_Tecnica_ES.md`). Section 3 explains the algorithm, section 5 the vector-search use case, and section 6.2 walks through this notebook section by section.

# %%
import subprocess, sys, importlib
for mod, spec in [("faiss", "faiss-cpu"), ("turbovec", "turbovec"), ("datasets", "datasets"), ("sklearn", "scikit-learn")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", spec], check=True)

# %% [markdown]
# ## The TurboQuant implementation
# Same reference module as the LLM notebook. For search we only need `TurboQuant`, `pack_bits` and `unpack_bits`.

# %%
# Needs turboquant_core.py in the same folder (in the notebook this cell writes it).

# %%
import math, time, os, gc
import numpy as np, pandas as pd, torch, faiss, turbovec
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print
import turboquant_core as tq
importlib.reload(tq)
from turboquant_core import TurboQuant, unpack_bits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| faiss", faiss.__version__, "| turbovec", turbovec.__version__, "| threads", faiss.omp_get_max_threads())

# %% [markdown]
# ## 1. Load embeddings

# %%
DATASET = "dbpedia-1536"      # "dbpedia-1536" | "dbpedia-3072" | "20newsgroups-lsa"
N_DB, N_Q = 100_000, 1_000

def load_dbpedia(dim):
    from datasets import load_dataset
    name = f"Qdrant/dbpedia-entities-openai3-text-embedding-3-large-{dim}-1M"
    ds = load_dataset(name, split="train", streaming=True)
    col, rows = None, []
    for r in ds:
        if col is None:   # find the embedding column (a long list of floats)
            col = next(k for k, v in r.items() if isinstance(v, (list, tuple)) and len(v) >= 256)
            print("embedding column:", col)
        rows.append(r[col])
        if len(rows) >= N_DB + N_Q:
            break
    return np.asarray(rows, dtype=np.float32)

def load_lsa(dim=384):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes")).data
    sents = [s.strip() for d in docs for s in d.replace("\n", " ").split(". ") if 40 < len(s.strip()) < 400]
    sents = list(dict.fromkeys(sents))[: N_DB + N_Q]
    X = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000).fit_transform(sents)
    return TruncatedSVD(dim, random_state=0).fit_transform(X).astype(np.float32)

def load_synthetic(dim=768, n_clusters=2000):
    """Last-resort offline data: anisotropic clustered Gaussians (clearly not real embeddings)."""
    g = np.random.default_rng(1)
    scales = (1.0 / np.arange(1, dim + 1) ** 0.5).astype(np.float32)
    centers = g.standard_normal((n_clusters, dim)).astype(np.float32) * scales
    lab = g.integers(0, n_clusters, N_DB + N_Q)
    return centers[lab] + 0.35 * g.standard_normal((N_DB + N_Q, dim)).astype(np.float32) * scales

t0 = time.time()
if DATASET.startswith("dbpedia"):
    try:
        E = load_dbpedia(int(DATASET.split("-")[1]))
    except Exception as e:
        print("Could not stream DBpedia (", e, ") - falling back to 20newsgroups LSA")
        DATASET = "20newsgroups-lsa"
if DATASET == "20newsgroups-lsa":
    try:
        E = load_lsa()
    except Exception as e:
        print("20newsgroups unavailable (", e, ") - using synthetic clustered data")
        DATASET, E = "synthetic-768", load_synthetic()
elif os.path.exists(DATASET):  # a local .npy file of embeddings also works
    E = np.load(DATASET).astype(np.float32)
DS_LABEL = os.path.basename(str(DATASET))
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12      # cosine similarity = inner product
rng = np.random.default_rng(0)
perm = rng.permutation(len(E))
n_db = min(N_DB, len(E) - N_Q)
X, Q = np.ascontiguousarray(E[perm[:n_db]]), np.ascontiguousarray(E[perm[n_db:n_db + N_Q]])
D = X.shape[1]
print(f"{DATASET}: database {X.shape}, queries {Q.shape}, loaded in {time.time() - t0:.0f}s, float32 size {X.nbytes / 2**20:.0f} MB")

# %% [markdown]
# ## 2. Ground truth and metrics
# * **Recall@1@k** (the paper's metric): how often the true nearest neighbour is inside the top-k returned.
# * **10@10**: overlap between the true top-10 and the returned top-10.

# %%
KMAX = 64
flat = faiss.IndexFlatIP(D); flat.add(X)
t = time.time(); _, GT = flat.search(Q, 100); t_flat = time.time() - t
KS = [1, 2, 4, 8, 16, 32, 64]

def recall_1_at_k(I):
    return {k: float(np.mean([GT[i, 0] in I[i, :k] for i in range(len(I))])) for k in KS}

def ten_at_ten(I):
    return float(np.mean([len(set(GT[i, :10]) & set(I[i, :10])) / 10 for i in range(len(I))]))

RESULTS = []
def record(name, family, bits, build_s, search_s, nbytes, I):
    r = dict(method=name, family=family, bits_per_dim=bits, build_s=round(build_s, 3),
             QPS=round(len(Q) / search_s), bytes_per_vec=round(nbytes / len(X), 1),
             compression=round(X.nbytes / nbytes, 1), ten_at_10=round(ten_at_ten(I), 4))
    r.update({f"R1@{k}": round(v, 4) for k, v in recall_1_at_k(I).items()})
    RESULTS.append(r)
    print(f"{name:34s} build {build_s:7.2f}s  QPS {r['QPS']:8d}  {r['compression']:5.1f}x smaller  R1@1 {r['R1@1']:.3f}  R1@8 {r['R1@8']:.3f}  10@10 {r['ten_at_10']:.3f}")

record("Exact float32 (FlatIP)", "exact", 32, 0.0, t_flat, X.nbytes, GT[:, :KMAX])

# %% [markdown]
# ## 3. TurboQuant, reference implementation
# Encoding = rotate, then pick the nearest of `2^b` fixed centroids per coordinate. No training, no codebook to learn.
# Search works in the rotated space: `<q, x̃> = ‖x‖ · <Πq, c[idx]>`, so we rotate each query once and score packed codes chunk by chunk.
# The `prod` variant adds the 1-bit QJL residual term, which makes the score an **unbiased** estimate of `<q, x>`.

# %%
class TurboQuantSearch:
    def __init__(self, d, bits, mode="mse", seed=0, device=DEVICE):
        self.q = TurboQuant(d, bits, mode=mode, seed=seed, device=device)
        self.device, self.mode, self.d = device, mode, d

    def add(self, X, batch=50_000):
        parts = [self.q.quantize(torch.from_numpy(X[i:i + batch]).to(self.device)) for i in range(0, len(X), batch)]
        self.codes = tq.Compressed(*(None if getattr(parts[0], f) is None else torch.cat([getattr(p, f) for p in parts])
                                     for f in ("idx", "norm", "qjl", "gamma")))
        return self

    def nbytes(self):
        return self.codes.nbytes()

    @torch.no_grad()
    def search(self, Q, k, chunk=25_000):
        q = torch.from_numpy(Q).to(self.device)
        qr = q @ self.q.Pi.T                                   # rotate queries once
        qs = (q @ self.q.S.T) if self.mode == "prod" else None
        c = self.codes
        best_s = torch.full((len(Q), k), -1e9, device=self.device); best_i = torch.zeros((len(Q), k), dtype=torch.long, device=self.device)
        for s in range(0, c.norm.shape[0], chunk):
            sl = slice(s, s + chunk)
            if self.q.mse_bits > 0:
                y = self.q.centroids[unpack_bits(c.idx[sl], self.q.mse_bits, self.d)]   # [chunk, d] rotated reconstruction
                score = qr @ y.T
            else:
                score = torch.zeros(len(Q), c.norm[sl].shape[0], device=self.device)
            if self.mode == "prod":
                z = unpack_bits(c.qjl[sl], 1, self.d).float() * 2 - 1
                score = score + (math.sqrt(math.pi / 2) / self.d) * (qs @ z.T) * c.gamma[sl].float()
            score = score * c.norm[sl].float()
            cs, ci = torch.cat([best_s, score], 1), torch.cat([best_i, torch.arange(s, s + score.shape[1], device=self.device).expand(len(Q), -1)], 1)
            best_s, top = cs.topk(k, dim=1)
            best_i = ci.gather(1, top)
        return best_s.cpu().numpy(), best_i.cpu().numpy()

for bits, mode in [(2, "mse"), (2, "prod"), (4, "mse"), (4, "prod")]:
    idx = TurboQuantSearch(D, bits, mode)
    t = time.time(); idx.add(X)
    if DEVICE == "cuda": torch.cuda.synchronize()
    tb = time.time() - t
    t = time.time(); _, I = idx.search(Q, KMAX); ts = time.time() - t
    record(f"TurboQuant_{mode} {bits}-bit (PyTorch)", "turboquant", bits, tb, ts, idx.nbytes(), I)
    del idx; gc.collect()

# %% [markdown]
# ## 4. turbovec: TurboQuant with SIMD kernels
# [`turbovec`](https://github.com/ryancodrai/turbovec) implements the same algorithm in Rust (AVX-512/AVX2/NEON). Same recall story, production-grade speed.

# %%
for bits in (2, 4):
    tv = turbovec.TurboQuantIndex(dim=D, bit_width=bits)
    t = time.time(); tv.add(X); tb = time.time() - t
    tv.prepare(); tv.search(Q[:8], 10)                        # warm-up
    t = time.time(); _, I = tv.search(Q, KMAX); ts = time.time() - t
    record(f"turbovec {bits}-bit", "turboquant", bits, tb, ts, len(tv.to_bytes()), np.asarray(I))
    del tv; gc.collect()

# %% [markdown]
# ## 5. Trained baselines (FAISS)
# PQ must learn codebooks with k-means before it can encode anything; the training time below is the "indexing time" the paper compares against. The paper's PQ uses 256 centroids per sub-space (LUT256): 4 dims per byte at 2 bits, 2 dims per byte at 4 bits.

# %%
TRAIN = X[rng.choice(len(X), min(25_000, len(X)), replace=False)]   # PQ k-means is the slow part; this takes a few minutes at d=1536

def run_faiss(name, family, bits, index, train=True):
    t = time.time()
    if train:
        index.train(TRAIN)
    index.add(X); tb = time.time() - t
    t = time.time(); _, I = index.search(Q, KMAX); ts = time.time() - t
    record(name, family, bits, tb, ts, index.sa_code_size() * len(X), I)

for bits in (2, 4):
    run_faiss(f"FAISS PQ {bits}-bit (LUT256)", "pq", bits, faiss.IndexPQ(D, D * bits // 8, 8, faiss.METRIC_INNER_PRODUCT))
    run_faiss(f"FAISS PQ-FastScan {bits}-bit", "pq", bits, faiss.IndexPQFastScan(D, D * bits // 4, 4, faiss.METRIC_INNER_PRODUCT))
    if hasattr(faiss, "IndexRaBitQ"):
        run_faiss(f"FAISS RaBitQ {bits}-bit", "rabitq", bits, faiss.IndexRaBitQ(D, faiss.METRIC_INNER_PRODUCT, bits))
run_faiss("FAISS SQ 4-bit (per-dim min/max)", "sq", 4, faiss.IndexScalarQuantizer(D, faiss.ScalarQuantizer.QT_4bit, faiss.METRIC_INNER_PRODUCT))

# %% [markdown]
# ## 6. Results

# %%
res = pd.DataFrame(RESULTS)
cols = ["method", "bits_per_dim", "compression", "bytes_per_vec", "build_s", "QPS", "R1@1", "R1@4", "R1@16", "R1@64", "ten_at_10"]
display(res[cols])

fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
style = {"turboquant": "-", "pq": "--", "rabitq": ":", "sq": "-.", "exact": "-"}
for bits, a in [(2, ax[0]), (4, ax[1])]:
    for _, r in res[(res.bits_per_dim == bits)].iterrows():
        a.plot(KS, [r[f"R1@{k}"] for k in KS], style[r.family], marker="o", ms=4, label=r.method)
    a.set(xscale="log", xticks=KS, xticklabels=KS, xlabel="top-k", ylabel="Recall@1@k", title=f"{DS_LABEL}: {bits}-bit")
    a.legend(fontsize=7); a.grid(alpha=.3)
q = res[res.family != "exact"]
ax[2].barh(q.method, q.build_s.clip(lower=1e-3), color=["C0" if f == "turboquant" else "C3" for f in q.family])
ax[2].set(xscale="log", xlabel="indexing time (s, log scale)", title="Indexing time (training + encoding)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7. Online ingestion
# Because TurboQuant needs no training, vectors can be added the moment they arrive, and the index never needs re-training as the data drifts. Here we stream the database in batches of 1,000.

# %%
tv = turbovec.TurboQuantIndex(dim=D, bit_width=4)
t = time.time()
for i in range(0, len(X), 1000):
    tv.add(X[i:i + 1000])
print(f"turbovec: streamed {len(tv):,} vectors in {time.time() - t:.2f}s with zero training")

# %% [markdown]
# ## 8. Two-stage search: compressed scan, then exact re-rank
# A common production pattern: scan the 8-16x smaller TurboQuant index for 100 candidates, then re-score those with full-precision vectors (kept on disk or in cheaper memory).

# %%
_, cand = tv.search(Q, 100)
cand = np.asarray(cand)
rer = np.stack([cand[i][np.argsort(-(X[cand[i]] @ Q[i]))] for i in range(len(Q))])
print(f"turbovec 4-bit top-100 then exact re-rank: R1@1 = {np.mean(rer[:, 0] == GT[:, 0]):.4f}, 10@10 = {ten_at_ten(rer):.4f}")

# %% [markdown]
# ### Takeaways
# * TurboQuant matches or beats trained PQ at the same bit budget, with **no training at all**: indexing is a matrix multiply plus a bucket lookup.
# * 4-bit TurboQuant keeps near-exact recall at 8x compression of float32; 2-bit gives 16x.
# * The `prod` (QJL) variant gives unbiased scores; the MSE variant usually ranks as well or better at 4 bits, as the paper's Figure 3 predicts.
# * With turbovec, the same algorithm runs with SIMD kernels, fast enough to replace PQ-FastScan in a real system.
