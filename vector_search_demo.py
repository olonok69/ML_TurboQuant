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
# **Companion reading:** the workshop guide (`docs/guide/TurboQuant_Technical_Guide_EN.md` in the repository; Spanish version `TurboQuant_Guia_Tecnica_ES.md`). Section 3 explains the algorithm, section 6 the vector-search use case, and section 7.2 walks through this notebook section by section.

# %% [markdown]
# ### Environment Setup
#
# Before running the notebook, we check for and install the required libraries: FAISS for vector search, `turbovec` for SIMD TurboQuant acceleration, Hugging Face `datasets` for streaming embeddings, and scikit-learn as a fallback dataset.

# %%
import subprocess, sys, importlib
for mod, spec in [("faiss", "faiss-cpu"), ("turbovec", "turbovec"), ("datasets", "datasets"), ("sklearn", "scikit-learn"), ("seaborn", "seaborn")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", spec], check=True)

# %% [markdown]
# ## The TurboQuant reference implementation
#
# Below, we write out the `turboquant_core.py` module. It contains:
# - Numeric Lloyd-Max codebook generation using cached 1D k-means.
# - High-performance bit-packing and unpacking algorithms.
# - The core `TurboQuant` class implementing Algorithm 1 (MSE) and Algorithm 2 (unbiased product estimation).
# - `MixedTurboQuant` for handling outlier split-channels.
# - A standard `UniformQuant` baseline for performance and compression comparison.

# %% [markdown]
# ### Summary of the `TurboQuant` Class
#
# The `TurboQuant` class is the core algorithm implementation for vector quantization. It offers two main modes of operation:
#
# 1. **`mode="mse"` (Algorithm 1):** Performs Mean Squared Error (MSE) quantization. It applies a Haar-random orthogonal rotation to the normalized input vector, then matches each coordinate to its nearest centroid in a fixed (pre-computed and cached) Lloyd-Max codebook.
# 2. **`mode="prod"` (Algorithm 2):** Adds a 1-bit QJL (Quantized Johnson-Lindenstrauss) projection to the quantization residual of the MSE stage. This provides an unbiased estimator for inner-product (cosine similarity) searches.
#
# Key features include:
# - **No Training Needed:** Runs out-of-the-box using data-oblivious random rotations instead of computationally heavy k-means clustering.
# - **Support for Arbitrary Vector Norms:** Norms are saved in half-precision float (`fp16`) for scale reconstruction during dequantization.
# - **Real Bit Packing:** Automatically packs quantized integer codes into compact `uint8` arrays, providing exact memory/disk savings rather than simulated estimations.

# %% [markdown]
# ### Implementation: `turboquant_core.py`
#
# The following code block writes the primary reference module containing the PyTorch implementations. It includes:
# 1. **Lloyd-Max Codebook Optimization**: Numerically finds centroid values for coordinate-wise scalar quantization using the standard probability density function of random unit vectors.
# 2. **Custom Bit-Packing/Unpacking**: Packs high-dimensional indices into raw `uint8` arrays to guarantee exact, measurable, zero-overhead storage reduction.
# 3. **The Core `TurboQuant` Class**: Houses both the standard minimum mean-squared-error (MSE) quantization and the unbiased Product (QJL residual-based) estimator modes.
# 4. **Mixed Precision Outlier Tracking**: Outliers with unusually high values are assigned higher quantization budgets dynamically through a calibration subset.
# 5. **Standard Baseline Comparisons**: A uniform vector-scale quantization module to run as a comparison.

# %%
# Needs turboquant_core.py in the same folder (in the notebook this cell writes it).

# %% [markdown]
# ### Dependencies, Module Reload, and System Diagnostics
#
# In this cell, we import all required libraries, reload our newly written `turboquant_core` module to ensure any code changes are applied, and check system diagnostic information including device availability (CPU/GPU), library versions, and maximum active threads.

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

# %% [markdown]
# ### Loading Dataset Embeddings
#
# In this section, we load the vector datasets used for testing. The code supports three main sources:
# 1. **DBpedia Entities (`dbpedia-1536` / `dbpedia-3072`)**: Streams high-dimensional embeddings generated with OpenAI's `text-embedding-3-large` models directly from Hugging Face.
# 2. **20 Newsgroups (`20newsgroups-lsa`)**: A local fallback dataset utilizing TF-IDF and Latent Semantic Analysis (LSA) dimensional reduction.
# 3. **Synthetic Clustered Data**: A localized random generation fallback using anisotropic clustered Gaussians if internet access or datasets are completely unavailable.
#
# All vectors are normalized to unit length so that cosine similarity can be computed via efficient inner-product operators.

# %% [markdown]
# ### Dataset Configuration and Loading Execution
#
# In this cell, we establish the configurations for the test run (`DATASET`, database size `N_DB`, and query size `N_Q`). We define the streaming loader function for the DBpedia dataset as well as offline fallbacks for 20 Newsgroups (LSA dimensionality reduced) and synthetic clustered Gaussian data. Finally, we execute the loading sequence and normalize the resulting arrays to unit length.

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
# ## 2. Ground Truth Generation and Evaluation Metrics
#
# In this section, we establish our ground-truth search results and define evaluation helper metrics:
# - **Exact float32 (FlatIP)**: Runs an exact search via FAISS `IndexFlatIP` to locate the exact top-100 nearest neighbors.
# - **Recall@1@k**: Measures how often the true top-1 (the absolute closest vector) is successfully retrieved within the top $k$ candidates returned by a quantized search.
# - **10@10 Overlap**: Measures the average precision overlap between the exact top-10 set and the returned top-10 set.
# - **record()**: A helper function to record compression ratios, indexing speed, QPS, and recall accuracy across different algorithms.

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
# ## 3. TurboQuant: Reference Implementation and Query Evaluation
#
# Encoding works by rotating the vectors with a random orthogonal matrix, then finding the nearest of $2^b$ precomputed Lloyd-Max centroids for each coordinate. No training or data clustering is needed.
#
# ### Code Search Logic (`TurboQuantSearch`):
# - **Initialization**: Creates a `TurboQuant` instance for vector rotation and centroid assignment.
# - **Adding Database Vectors**: Quantizes vectors in large batches (e.g., 50k at a time) and concatenates their packed representations to minimize CPU overhead.
# - **Search Execution**:
#   - Rotates each query vector once up front.
#   - Processes compressed database blocks in chunks to keep memory consumption low.
#   - Unpacks the bit-packed representation on the fly, scores the rotated inputs, and optionally adds the unbiased 1-bit QJL residual term for `prod` mode.
#   - Tracks the running nearest-neighbor candidates across blocks using PyTorch `topk` on the current device.

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
#
# [`turbovec`](https://github.com/ryancodrai/turbovec) is a production-grade, highly accelerated Rust implementation of the TurboQuant algorithm supporting hardware-specific instruction sets (AVX-512, AVX2, and ARM NEON).
#
# ### Integration and Execution Flow:
# - **Index Creation**: Initializes the SIMD index with the target dimension and the specific bit budget (2 or 4 bits).
# - **Asynchronous Ingestion**: Measures database population time using `add()`, converting native Python matrices into the highly compressed, byte-aligned physical structures within Rust.
# - **Search Execution**: Performs a warm-up lookup to trigger JIT optimization paths, runs queries against the database, and measures execution throughput (Queries Per Second) with high-efficiency memory management.

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
#
# Unlike TurboQuant, which uses data-oblivious random rotations, standard Product Quantization (PQ) baselines require a computationally expensive training phase (using k-means clustering) to learn representative codebooks before encoding vectors.
#
# ### Executed Baselines:
# 1. **FAISS PQ (LUT256)**: The paper's baseline, partitioning the vector space into sub-spaces with 256 centroids each (8 bits per sub-space).
# 2. **FAISS PQ FastScan**: Optimized for high speed by restricting sub-spaces to 16 centroids (4 bits) to enable SIMD-based register-level lookup tables.
# 3. **FAISS RaBitQ**: A robust quantization baseline designed for inner-product space indexing.
# 4. **FAISS SQ4 (Scalar Quantizer)**: A simple baseline performing uniform 4-bit min-max scaling across each individual coordinate dimension.

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
#
# This section consolidates the performance statistics of all tested quantizers and visualizes their trade-offs across three dimensions:
#
# ### Metric Table:
# - Gathers indexing build times (seconds), query throughput (Queries Per Second, QPS), exact memory usage per vector (bytes), compression ratio, and recall accuracy across a range of neighborhood depths ($k$).
#
# ### Visualization Plots:
# - **Recall curves (2-bit & 4-bit)**: Compares Recall@1@$k$ for varying candidate set sizes ($k$). Flat lines near $1.0$ indicate higher precision.
# - **Indexing Time Comparison**: A horizontal bar chart contrasting the overall build time (including training and database encoding) of all families. Shorter bars represent faster indexing.

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
# ### 6c. Unified Speed (QPS) vs. Accuracy (Recall@1) Comparison
#
# This visualization directly plots **Query Throughput (QPS)** against **Retrieval Accuracy (Recall@1)** for every single search configuration recorded in our experiment.
#
# ### How to Interpret This Plot:
# - **Top-Right is Optimal**: The ideal vector search algorithm resides in the upper-right corner of this coordinate space, meaning it delivers near-perfect retrieval accuracy (Recall@1 $\approx 1.0$) while processing queries at extremely high throughput rates.
# - **Logarithmic Scale**: The Y-axis (QPS) is plotted on a logarithmic scale to properly contrast highly accelerated SIMD engines like `turbovec` against pure Python reference loops or slower memory-bound lookup implementations.
# - **Method Comparison**: This allows us to inspect at a glance how the data-oblivious **TurboQuant** methods match or exceed both the speed and accuracy trade-offs of heavily trained alternatives (such as FAISS PQ and RaBitQ) across different bit budgets.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

try:
    # Prepare dataset from the results
    df_qps_rec = pd.DataFrame(RESULTS)

    plt.figure(figsize=(10, 6.5))

    # Plot each method with a distinct marker and color mapped by its family category
    sns.scatterplot(
        data=df_qps_rec,
        x="R1@1",
        y="QPS",
        hue="family",
        style="family",
        s=150,
        edgecolor="black",
        alpha=0.9
    )

    # Add labels to every point on the plot
    for _, row in df_qps_rec.iterrows():
        label = row["method"].replace(" (PyTorch)", "")
        plt.text(
            row["R1@1"] + 0.002,
            row["QPS"] * 1.05,
            label,
            fontsize=8.5,
            alpha=0.85
        )

    plt.yscale("log")
    plt.xlim(df_qps_rec["R1@1"].min() - 0.02, 1.02)
    plt.title("QPS vs. Recall@1 for All Evaluation Methods", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Recall @ 1 @ 1 (Accuracy)", fontsize=11)
    plt.ylabel("Queries Per Second (QPS - Log Scale)", fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title="Quantizer Family", loc="lower left", frameon=True)
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Error generating QPS vs Recall plot:", e)

# %% [markdown]
# ### 6b. Deep-Dive Performance Trade-off Visualization
#
# To better understand the operational efficiency of **TurboQuant** and `turbovec` compared to standard FAISS baselines, we plot two multi-attribute trade-off spaces:
#
# 1. **Query Throughput (QPS) vs. Accuracy (Recall@1)**:
#    - **How to read**: The optimal region is the **top-right** (high accuracy, high speed).
#    - **What to look for**: Notice how `turbovec` clusters near high-performance throughput regimes while matching or exceeding the accuracy of data-trained alternatives.
#
# 2. **Indexing Build Time vs. Storage Compression**:
#    - **How to read**: The optimal region is the **top-left** (ultra-fast index build, maximum memory compression).
#    - **What to look for**: See how the training-free design of TurboQuant places it far to the left of k-means clustering techniques (like FAISS PQ), creating index builds that are orders of magnitude faster without sacrificing compression quality.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

try:
    # Prepare dataset for visualization
    df_plot = pd.DataFrame(RESULTS)

    # Setup figure canvas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color palette map for consistent labeling
    families = df_plot['family'].unique()
    colors = sns.color_palette('Set2', len(families))
    color_map = dict(zip(families, colors))

    # 1. QPS vs. Recall@1
    for family, group in df_plot.groupby('family'):
        ax1.scatter(
            group['R1@1'],
            group['QPS'],
            label=family.upper(),
            color=color_map[family],
            s=120,
            edgecolors='black',
            alpha=0.85
        )
        for _, row in group.iterrows():
            ax1.annotate(
                row['method'].replace(' (PyTorch)', ''),
                (row['R1@1'], row['QPS']),
                textcoords='offset points',
                xytext=(5, 5),
                fontsize=8
            )

    ax1.set_yscale('log')
    ax1.set_xlabel('Recall @ 1 @ 1 (Accuracy)', fontsize=11)
    ax1.set_ylabel('Queries Per Second (QPS - Log Scale)', fontsize=11)
    ax1.set_title('Query Speed (QPS) vs. Recall Accuracy Trade-off (Top-Right is Best)', fontsize=12, pad=15)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(title='Quantizer Family', loc='best')

    # 2. Build Time vs. Compression Ratio
    # Filter out exact baseline for indexing time plots to avoid inf/zero skew
    df_indexed = df_plot[df_plot['family'] != 'exact'].copy()

    for family, group in df_indexed.groupby('family'):
        # Use np.clip to safely limit very fast build times (like FlatIP or zero-seconds)
        build_times = np.clip(group['build_s'].astype(float).values, 1e-3, None)
        ax2.scatter(
            build_times,
            group['compression'],
            label=family.upper(),
            color=color_map[family],
            s=120,
            edgecolors='black',
            alpha=0.85
        )
        for idx_row, row in enumerate(group.itertuples()):
            ax2.annotate(
                row.method.replace(' (PyTorch)', ''),
                (build_times[idx_row], row.compression),
                textcoords='offset points',
                xytext=(5, 5),
                fontsize=8
            )

    ax2.set_xscale('log')
    ax2.set_xlabel('Indexing Build Time (Seconds - Log Scale)', fontsize=11)
    ax2.set_ylabel('Memory Compression Ratio (x times smaller)', fontsize=11)
    ax2.set_title('Index Build Speed vs. Compression Ratio (Top-Left is Best)', fontsize=12, pad=15)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(title='Quantizer Family', loc='best')

    plt.tight_layout()
    plt.show()
except Exception as e:
    import traceback
    traceback.print_exc()

# %% [markdown]
# ### 6d. Family-Level Performance Aggregation (Average QPS)
#
# To see the high-level efficiency of each indexing framework, we can group the individual methods by their algorithm **family** (e.g., `turboquant`, `pq`, `rabitq`, `sq`) and compute the average **Queries Per Second (QPS)**.
#
# - **Interpretation**: This calculation helps contrast the performance throughput across the entire family, showing the real-world impact of hardware-friendly optimizations (such as `turbovec`'s SIMD execution) compared to standard algorithmic families.

# %%
import pandas as pd

try:
    # Load the results list into a DataFrame
    df_res = pd.DataFrame(RESULTS)

    # Group by 'family' and calculate the average QPS
    avg_qps_by_family = df_res.groupby('family')['QPS'].mean().reset_index()

    # Sort the values from highest to lowest QPS for better readability
    avg_qps_by_family = avg_qps_by_family.sort_values(by='QPS', ascending=False).reset_index(drop=True)

    # Display the results table
    print("Average QPS by Quantizer Family:")
    display(avg_qps_by_family)
except Exception as e:
    print("Error computing average QPS by family:", e)

# %% [markdown]
# ### Summary Report: Top-Performing Methods by QPS
#
# Based on the experimental runs recorded on the `dbpedia-1536` dataset ($100k$ database vectors, $1k$ query vectors), here is a summary ranking the top-performing individual vector search configurations by their **Queries Per Second (QPS)**.
#
# #### 1. Top Performing Configurations
#
# | Rank | Method / Configuration | Family | Bits/Dim | QPS | Recall @ 1 | Compression | Indexing Build Time |
# | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
# | **1** | `turbovec 2-bit` | `turboquant` | 2 | **305** | 0.892 | 15.5x | 3.25s |
# | **2** | `FAISS PQ-FastScan 2-bit` | `pq` | 2 | **246** | 0.882 | 16.0x | 16.15s |
# | **3** | `Exact float32 (FlatIP)` | `exact` | 32 | **180** | 1.000 | 1.0x | 0.00s (No build) |
# | **4** | `FAISS PQ-FastScan 4-bit` | `pq` | 4 | **180** | 0.959 | 8.0x | 49.32s |
# | **5** | `turbovec 4-bit` | `turboquant` | 4 | **126** | 0.971 | 7.8x | 2.37s |
#
# #### 2. Key Analysis & Insights
#
# * **SIMD Hardware Acceleration is King:** The fastest search engine evaluated is **`turbovec 2-bit` (305 QPS)**. By utilizing custom compiled Rust SIMD register lookups, it outperforms all other quantized baselines (including FAISS's highly optimized PQ-FastScan registers) while achieving a near-zero index build time ($3.25$ seconds) with zero training phase.
# * **The Index Build Time Bottleneck:** Standard Product Quantization (`FAISS PQ-FastScan`) achieves high operational throughput but requires a computationally heavy training phase via k-means. For example, building the `FAISS PQ-FastScan 4-bit` index takes **$49.32$ seconds** compared to only **$2.37$ seconds** for `turbovec 4-bit`—making `turbovec` over **20x faster to index** with superior retrieval accuracy (0.971 vs 0.959 Recall@1).
# * **Pure Python vs. Compiled Kernels:** The reference PyTorch implementations (`TurboQuant_mse 2-bit`) perform well mathematically but are bounded by Python loop interpreter overheads (~$70$ QPS), highlighting why compiled Rust/C++ SIMD layers (such as `turbovec`) are critical for real-world high-throughput production deployment.

# %% [markdown]
# ### 6e. R1@4 Recall Performance Comparison
#
# To understand retrieval quality with very tight candidate budgets, we can isolate and compare the **Recall@1@4** (the probability that the true top-1 item is found within the top-4 retrieved candidates) across all configurations. This metric is crucial for systems that use a small multi-stage re-ranking pool.

# %%
import pandas as pd

try:
    # Load results into a DataFrame
    df_recall = pd.DataFrame(RESULTS)

    # Select and rename columns for clarity
    recall_comparison = df_recall[['method', 'family', 'bits_per_dim', 'R1@4', 'compression']].copy()
    recall_comparison = recall_comparison.sort_values(by='R1@4', ascending=False).reset_index(drop=True)

    print("Recall@1@4 Comparison Across All Methods (Sorted Highest to Lowest):")
    display(recall_comparison)
except Exception as e:
    print("Error computing R1@4 recall comparison:", e)

# %% [markdown]
# ## 7. Online ingestion
#
# Because TurboQuant uses data-oblivious random orthogonal rotations instead of data-dependent cluster centroid training, it completely eliminates the index retraining phase required when database contents change.
#
# ### Ingestion Workflow:
# - **Streaming Additions**: Demonstrates inserting vectors on-the-fly in dynamic batches of 1,000.
# - **Zero-Retraining Cost**: Shows how the underlying database scales immediately without requiring recalculations of global quantizer centroids or dealing with data drift degradation.

# %%
tv = turbovec.TurboQuantIndex(dim=D, bit_width=4)
t = time.time()
for i in range(0, len(X), 1000):
    tv.add(X[i:i + 1000])
print(f"turbovec: streamed {len(tv):,} vectors in {time.time() - t:.2f}s with zero training")

# %% [markdown]
# ## 8. Two-stage search: compressed scan, then exact re-rank
#
# This section demonstrates a powerful system design pattern: scanning a highly-compressed, memory-resident index to find candidates, then performing exact full-precision distance calculations only on those candidates.
#
# ### Two-Stage Pipeline:
# 1. **Coarse Compressed Scan**: Scans the 8x to 16x smaller TurboQuant index to quickly filter out the top 100 closest candidate matches.
# 2. **Fine Full-Precision Re-rank**: Pulls the raw `float32` vectors of only those 100 candidate IDs (which can be stored on slower, cheaper disk memory) and calculates exact inner products to retrieve the final accurate rankings.

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
