# %% [markdown]
# # TurboQuant KV-cache compression: LLM inference demo (Google Colab)
#
# This notebook reproduces the core ideas of **TurboQuant** (Zandieh, Daliri, Hadian, Mirrokni, Google Research, arXiv:2504.19874)
# end-to-end in pure PyTorch, on the model the paper evaluates.
#
# 1. **Theory check.** TurboQuant's measured distortion versus the paper's upper and lower bounds.
# 2. **Quality.** Perplexity, KL divergence and top-1 agreement versus the FP16 cache, on WikiText-2.
# 3. **Needle in a haystack.** Long-context retrieval with the compressed cache.
# 4. **Memory.** Measured KV-cache bytes and GPU memory for an 8B model.
# 5. **Speed.** Why fused kernels matter (e.g. vLLM).
#
# **Why Llama-3.1-8B-Instruct?**
# It is the model the paper uses for its LongBench and needle-in-a-haystack results (Table 1 and Figure 4), so the numbers here can be read next to the paper's.
# An earlier version of this demo used Qwen2.5-1.5B-Instruct on a free T4. In our test runs, smaller models such as Qwen2.5 and Llama-2 lost the needle once the cache was compressed to low bit-widths.
# The next cell prints Llama's attention code: Hugging Face applies RoPE to the keys *before* `past_key_values.update(...)`, so the cache always stores rotated (post-RoPE) keys.
# That is the same for every model in `transformers` and for the paper's own setup, so RoPE alone does not explain the difference.
# The likely cause is that Qwen2 models add a bias to `k_proj` and have a few key channels with very large values, which low-bit quantizers handle badly; Llama-3.1's keys are better behaved.
# (This explanation is inferred; the Qwen runs were not kept.)
#
# **Runtime:** `Runtime > Change runtime type > A100` (40 GB) or **L4** (24 GB). The model needs about 16 GB in FP16, so it does not fit a free T4.
# It is a gated model: accept Meta's licence on its Hugging Face page, then add your token as a Colab secret named `HF_TOKEN`.
#
# > The reference implementation stores keys and values bit-packed and dequantizes them on read. That saves memory but adds work, so it is **slower** than the FP16 cache. Production speedups need fused kernels (vLLM ships them: `--kv-cache-dtype turboquant_4bit_nc`).
#
# **Companion reading:** the workshop guide (`docs/guide/TurboQuant_Technical_Guide_EN.md` in the repository; Spanish version `TurboQuant_Guia_Tecnica_ES.md`). Section 3 explains the algorithm, section 5 the KV-cache use case, and section 7.1 walks through this notebook section by section.

# %%
# Installs what Colab lacks; transformers must be v5 for the custom cache.
import importlib, subprocess, sys
def _ensure(pkg, spec):
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", spec], check=True)
import transformers
if int(transformers.__version__.split(".")[0]) < 5:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", "transformers>=5.0,<6", "accelerate"], check=True)
    print("Upgraded transformers. Please restart the runtime (Runtime > Restart session) and run again.")
for p, s in [("datasets", "datasets"), ("matplotlib", "matplotlib"), ("pandas", "pandas"), ("seaborn", "seaborn")]:
    _ensure(p, s)
print("transformers", transformers.__version__)

# %% [markdown]
# ### Where the KV cache sits in Llama's attention
# The printout below shows that `apply_rotary_pos_emb` runs before `past_key_values.update(...)`: the cache receives keys that already carry RoPE.

# %%
import inspect
from transformers.models.llama.modeling_llama import LlamaAttention
print("Full LlamaAttention forward pass:")
print(inspect.getsource(LlamaAttention.forward))

# %% [markdown]
# ## The TurboQuant implementation
# The next cell writes `turboquant_core.py` (about 400 lines, readable). Below is the complete architectural breakdown of the vector quantization pipeline implemented in this module:
#
# ### 1. Optimal Lloyd-Max Codebooks
# * `lloyd_max_codebook(d, bits)`: Calculates the mathematically optimal scalar centroids for the coordinates of a uniformly random unit vector of dimension $d$. Because these coordinates follow a scaled Beta distribution (as proved in Lemma 1 of the paper), this function uses 1D k-means to compute the optimal quantizer centroids and caches them via `@lru_cache` for speed.
#
# ### 2. Custom Coordinate-Wise Bit Packing
# * `pack_bits` & `unpack_bits`: Standard PyTorch works with 8-bit bytes at minimum. To measure true physical memory footprints for non-standard bit widths (like 3-bit or 1-bit flags), these functions perform bitwise shift operations to pack smaller arbitrary-width integer codes tightly into dense `uint8` tensors.
#
# ### 3. TurboQuant Quantizers
# * **Haar-Random Orthogonal Rotation (`random_rotation`)**: Applies a Householder-based QR decomposition to generate a reproducible, seed-based orthogonal matrix $\Pi$ used to spread outlier channels evenly.
# * **`TurboQuant` Class**:
#   * **`quantize`**: Computes the vector's Euclidean norm, normalizes it to a unit vector, rotates it using the orthogonal matrix $\Pi$, maps the coordinates to their closest Lloyd-Max centroids, and packs the bits.
#   * **`dequantize`**: Unpacks the codes, maps them back to centroids, applies the inverse rotation, and scales the output back by the stored norm.
# * **`MixedTurboQuant` Class**: Realizes the outlier split (e.g., 2.5-bit or 3.5-bit configurations). It dynamically identifies outlier indices (dimensions with the highest average magnitude) during calibration, and handles high-precision bits for outliers while keeping low-precision bits for standard dimensions.
# * **`UniformQuant` Class**: Standard asymmetric min-max integer quantization that stores block-level scale and zero parameters for comparison.
#
# ### 4. Integration with Hugging Face (`TurboQuantCache`)
# * **`TurboQuantLayer`**: Inherits from Hugging Face's `DynamicLayer`. When a sequence generates new tokens, they are initially processed in full precision. Once they exceed a specified `residual_length`, the historical keys and values are compressed on the fly and appended to a growing, compressed cache.
# * **`TurboQuantCache`**: Coordinates these compressed layers to act as a drop-in replacement for standard key-value storage caches in Hugging Face generation loops.

# %%
# Needs turboquant_core.py in the same folder (in the notebook this cell writes it).

# %%
import math, time, gc, random
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ImportError:
    display = print
import importlib, turboquant_core as tq
importlib.reload(tq)
from turboquant_core import (TurboQuant, MixedTurboQuant, UniformQuant, TurboQuantCache, fp16_cache_nbytes,
                             tq_factory, mixed_factory, uniform_factory, codebook_mse_cost, lloyd_max_codebook)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
torch.manual_seed(0); np.random.seed(0); random.seed(0)

# %% [markdown]
# ## 1. Theory check: distortion versus the paper's bounds
# For unit vectors, Theorem 1 says `D_mse <= (sqrt(3)*pi/2) * 4^-b` and Theorem 3 says no quantizer can beat `4^-b`.
# The paper reports `D_mse ≈ 0.36, 0.117, 0.03, 0.009` for b = 1..4.
#
# ### Code Explanation:
# 1. **Data Generation**: It creates random normal unit vectors $x$ and $y$ of dimension $d = 128$ in PyTorch. The vector $y$ (representing queries) is correlated with $x$ (representing keys) so their inner products are non-trivial.
# 2. **Theoretical Validation Loop**: For bit widths $b \in [1, 5]$, it quantizes and dequantizes (roundtrip) the vectors using both the MSE-optimized variant (`TurboQuant_mse`) and the unbiased variant (`TurboQuant_prod`). It measures actual Mean Squared Error (MSE), compares it against the paper's theoretical bounds, and tracks the bias and error of the reconstructed inner products.
# 3. **Visualizations** (using `matplotlib`):
#    - **Subplot 1**: Plots the measured MSE against the information-theoretic lower and upper bounds, showing that the empirical distortion falls perfectly within the predicted limits.
#    - **Subplot 2**: Plots a histogram of the error in inner products at 2 bits, demonstrating how the standard MSE variant is biased (shifted left) while the product variant (`TurboQuant_prod`) remains centered and unbiased.
#    - **Subplot 3**: Illustrates that the coordinates of randomly rotated unit vectors converge to a standard normal distribution, allowing a single precomputed Lloyd-Max codebook to work universally.

# %%
d = 128   # a typical attention head dimension
N = 20000
x = torch.randn(N, d, device=DEVICE); x = x / x.norm(dim=1, keepdim=True)
y = torch.randn(N, d, device=DEVICE); y = y / y.norm(dim=1, keepdim=True)
y = (0.6 * x + 0.8 * y)  # queries correlated with the data, so inner products are not ~0
rows = []
for b in range(1, 6):
    q_mse = TurboQuant(d, b, "mse", seed=1, device=DEVICE)
    q_prod = TurboQuant(d, b, "prod", seed=1, device=DEVICE)
    xm, xp = q_mse.roundtrip(x), q_prod.roundtrip(x)
    ip, ipm, ipp = (y * x).sum(1), (y * xm).sum(1), (y * xp).sum(1)
    rows.append(dict(bits=b,
                     mse_measured=((x - xm) ** 2).sum(1).mean().item(),
                     mse_theory=codebook_mse_cost(d, b),
                     lower_bound=4.0 ** -b,
                     upper_bound=math.sqrt(3) * math.pi / 2 * 4.0 ** -b,
                     ip_bias_mse=((ipm - ip).mean() / ip.mean()).item(),
                     ip_bias_prod=((ipp - ip).mean() / ip.mean()).item(),
                     ip_err_mse_x_d=(((ipm - ip) ** 2).mean() * d).item(),
                     ip_err_prod_x_d=(((ipp - ip) ** 2).mean() * d).item()))
theory = pd.DataFrame(rows)
display(theory.round(4))

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].semilogy(theory.bits, theory.mse_measured, "o-", label="TurboQuant_mse (measured)")
ax[0].semilogy(theory.bits, theory.upper_bound, "--", label="upper bound  √3π/2·4^-b")
ax[0].semilogy(theory.bits, theory.lower_bound, ":", label="lower bound  4^-b")
ax[0].set(xlabel="bits per coordinate", ylabel="MSE", title="MSE vs. information-theoretic bounds"); ax[0].legend()
q1m, q1p = TurboQuant(d, 2, "mse", seed=3, device=DEVICE), TurboQuant(d, 2, "prod", seed=3, device=DEVICE)
ax[1].hist(((y * q1m.roundtrip(x)).sum(1) - (y * x).sum(1)).cpu(), 100, alpha=.6, label="TurboQuant_mse (biased)")
ax[1].hist(((y * q1p.roundtrip(x)).sum(1) - (y * x).sum(1)).cpu(), 100, alpha=.6, label="TurboQuant_prod (unbiased)")
ax[1].axvline(0, c="k", lw=1); ax[1].set(title="Inner-product error at 2 bits", xlabel="<y,x̃> - <y,x>"); ax[1].legend()
c = lloyd_max_codebook(d, 3) * math.sqrt(d)
grid = np.linspace(-4, 4, 400)
ax[2].plot(grid, np.exp(-grid ** 2 / 2) / math.sqrt(2 * math.pi), label="coordinate density after rotation (×√d)")
ax[2].vlines(c, 0, 0.42, colors="C3", label="3-bit Lloyd-Max centroids")
ax[2].set(title="Rotated coordinates are ~N(0,1/d): one codebook fits all"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Load a model
# We use `meta-llama/Llama-3.1-8B-Instruct` (32 layers, 8 KV heads, head_dim 128), which is the exact model evaluated in the original paper.
# Head dim 128 lets us use exactly the paper's outlier split: 32 channels at 3 bits + 96 at 2 bits, which the paper calls **2.5 bits**. (The codes alone are 2.25 bits/channel; with the two fp16 norms the measured cost is exactly 2.5. Likewise the 3.5-bit split, 64 @4b + 64 @3b, measures 3.75 with norms. The `bits_per_channel` column below is always measured from the packed tensors.)
#
# ### Explanation of the Loading and Setup Steps:
# 1. **Model & Tokenizer Loading**: We use Hugging Face's `AutoTokenizer` and `AutoModelForCausalLM` to load the 8B model. (Requires HF Token authentication).
# 2. **Configuration Dictionary (`CONFIGS`)**: We register all KV-cache evaluation setups:
#    - **Uncompressed (Baseline)**: Normal FP16 cache representation.
#    - **TurboQuant ($b$-bit)**: Standard MSE quantization using `tq_factory` for 4, 3, and 2 bits.
#    - **TurboQuant Outlier Splits (3.5-bit & 2.5-bit)**: Mixed-precision layouts where standard/outlier dimensions are calibrated separately.
#    - **TurboQuant_prod**: Evaluates unbiased inner products using 2-bit MSE + 1-bit QJL residuals.
#    - **Uniform INT Baseline**: Standard uniform integer quantization with per-group scales and zero points to serve as a classic benchmark.
# 3. **Helper Functions**: Defines functions to instantiate caches, calculate actual KV-cache footprint in bytes, and measure the effective bits per channel dynamically.

# %%
# =====================================================================
# Model Setup and Quantization Configuration Definitions
# =====================================================================
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch

try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass  # Allow fallback to standard huggingface-cli login if secret is missing

# Using Llama-3.1-8B-Instruct as used in the original paper.
# Note: This is a gated model. You must accept the agreement on Hugging Face and provide an HF_TOKEN.
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" if DEVICE == "cuda" else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 1. Load Tokenizer and Base Model in half/full precision depending on Hardware
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
cfg = model.config
HEAD_DIM = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
print(MODEL_ID, "| layers", cfg.num_hidden_layers, "| kv heads", cfg.num_key_value_heads, "| head_dim", HEAD_DIM)

# 2. Define configurations to map names to respective quantizer factories
CONFIGS = {
    "Uncompressed cache (baseline)":            None,
    "TurboQuant 4-bit":                 (tq_factory(4), tq_factory(4)),
    "TurboQuant 3.5-bit (outlier split)": (mixed_factory(0.5, 4, 3), mixed_factory(0.5, 4, 3)),
    "TurboQuant 3-bit":                 (tq_factory(3), tq_factory(3)),
    "TurboQuant 2.5-bit (outlier split)": (mixed_factory(0.25, 3, 2), mixed_factory(0.25, 3, 2)),
    "TurboQuant 2-bit":                 (tq_factory(2), tq_factory(2)),
    "TurboQuant_prod 3-bit keys (MSE 2b + QJL 1b)": (tq_factory(3, "prod"), tq_factory(3)),
    "INT4 + scale/zero per 32 ch":      (uniform_factory(4), uniform_factory(4)),
    "INT3 + scale/zero per 32 ch":      (uniform_factory(3), uniform_factory(3)),
    "INT2 + scale/zero per 32 ch":      (uniform_factory(2), uniform_factory(2)),
}

# 3. Dynamic Cache Factories and Metric Utilities
def make_cache(name, residual_length=0):
    spec = CONFIGS[name]
    if spec is None:
        return DynamicCache(config=cfg)
    return TurboQuantCache(cfg, spec[0], spec[1], residual_length=residual_length)

def cache_nbytes(cache):
    return cache.nbytes() if isinstance(cache, TurboQuantCache) else fp16_cache_nbytes(cache)

def bits_per_channel(cache, n_tokens):
    n_vec = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * n_tokens
    return 8 * cache_nbytes(cache) / (n_vec * HEAD_DIM)

# %% [markdown]
# ## 3. Quality: perplexity and agreement with the FP16 model
# Text is fed in chunks of 128 tokens. Each chunk attends exactly to itself and to the **compressed** cache of everything before it, which mimics streaming generation where every past token lives in the quantized cache.
# The text is the WikiText-2 test set. If it cannot be downloaded, the cell falls back to a repeated sentence: perplexity is then close to 1 for every configuration and only the KL and top-1 columns are meaningful.
# Metrics against the FP16 run: perplexity, mean KL(FP16 ‖ quantized) of the next-token distribution, and top-1 agreement.
#
# ### Explanation of the Evaluation Metrics and Setup:
# 1. **Data Stream Simulation**: Instead of loading the entire sequence at once, we feed text in successive chunks of size `CHUNK = 128`. This enforces realistic autoregressive attention where historical context relies purely on decoded/dequantized KV tensors.
# 2. **Perplexity (PPL)**: Measures how well the compressed cache predicts the ground-truth sequence. A large jump in perplexity indicates information loss in the quantized keys/values.
# 3. **Kullback-Leibler (KL) Divergence**: Evaluates the statistical distance between the uncompressed next-token probability distribution (baseline) and the quantized model's distribution. Lower is better.
# 4. **Top-1 Agreement**: Represents the exact match rate of the highest-probability token predicted by the baseline vs. the quantized variant, showing how quantization impacts immediate greedy decisions.

# %%
# =====================================================================
# Quality Evaluation: Perplexity, KL Divergence, and Token Agreement
# =====================================================================
from datasets import load_dataset
import importlib
import gc, time, math
import torch
import pandas as pd
import turboquant_core as tq

# Force reload the corrected turboquant_core to ensure quality metrics use our fixes
importlib.reload(tq)
from turboquant_core import (
    TurboQuant, MixedTurboQuant, UniformQuant, TurboQuantCache, fp16_cache_nbytes,
    tq_factory, mixed_factory, uniform_factory
)

# Re-bind configurations to use reloaded factories
CONFIGS = {
    "Uncompressed cache (baseline)":            None,
    "TurboQuant 4-bit":                 (tq_factory(4), tq_factory(4)),
    "TurboQuant 3.5-bit (outlier split)": (mixed_factory(0.5, 4, 3), mixed_factory(0.5, 4, 3)),
    "TurboQuant 3-bit":                 (tq_factory(3), tq_factory(3)),
    "TurboQuant 2.5-bit (outlier split)": (mixed_factory(0.25, 3, 2), mixed_factory(0.25, 3, 2)),
    "TurboQuant 2-bit":                 (tq_factory(2), tq_factory(2)),
    "TurboQuant_prod 3-bit keys (MSE 2b + QJL 1b)": (tq_factory(3, "prod"), tq_factory(3)),
    "INT4 + scale/zero per 32 ch":      (uniform_factory(4), uniform_factory(4)),
    "INT3 + scale/zero per 32 ch":      (uniform_factory(3), uniform_factory(3)),
    "INT2 + scale/zero per 32 ch":      (uniform_factory(2), uniform_factory(2)),
}

try:
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")   # the bare "wikitext" id no longer resolves
    text = "\n".join(t for t in wiki["text"] if t.strip())
except Exception as e:
    print("wikitext unavailable, using a built-in passage:", e)
    text = ("The history of computing is a story of compression. " * 400)

N_TOK = 2048 if DEVICE == "cuda" else 512
CHUNK = 128
ids = tok(text, return_tensors="pt").input_ids[:, :N_TOK].to(DEVICE)

@torch.no_grad()
def chunked_logits(cache_name):
    cache = make_cache(cache_name)
    out = []
    for s in range(0, ids.shape[1], CHUNK):
        out.append(model(ids[:, s:s + CHUNK], past_key_values=cache, use_cache=True).logits.float())
    return torch.cat(out, 1), cache

ref_logits, ref_cache = chunked_logits("Uncompressed cache (baseline)")
ref_logp = torch.log_softmax(ref_logits[0, :-1], -1)
targets = ids[0, 1:]
BASE_BITS = bits_per_channel(ref_cache, ids.shape[1])
results = []

for name in CONFIGS:
    t0 = time.time()
    logits, cache = (ref_logits, ref_cache) if CONFIGS[name] is None else chunked_logits(name)
    logp = torch.log_softmax(logits[0, :-1], -1)

    nll = -logp.gather(-1, targets[:, None]).mean().item()
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1).mean().item()
    agree = (logp.argmax(-1) == ref_logp.argmax(-1)).float().mean().item()

    results.append(dict(
        config=name,
        bits_per_channel=round(bits_per_channel(cache, ids.shape[1]), 3),
        compression=round(BASE_BITS / bits_per_channel(cache, ids.shape[1]), 2),
        perplexity=round(math.exp(nll), 3),
        KL_vs_baseline=round(kl, 5),
        top1_agree=round(agree, 4),
        kv_MB=round(cache_nbytes(cache) / 2 ** 20, 2)
    ))
    del cache; gc.collect()

quality = pd.DataFrame(results)
display(quality)

# %% [markdown]
# ### Explanation of the Visualization (Perplexity vs. Memory & Divergence):
# 1. **Subplots Setup**: The code initializes a 1x2 subplot grid showing two crucial memory-vs-quality curves side-by-side.
# 2. **Color Coding**: Configurations are systematically color-coded to visually distinguish approaches:
#    - **Blue (`C0`)** represents **TurboQuant** configurations.
#    - **Red (`C3`)** represents the **Uniform INT** baselines.
#    - **Black (`k`)** represents the uncompressed **FP16 baseline**.
# 3. **Left Plot (Perplexity vs. Effective Bits)**: Compares the compression rate (effective bits per channel) against language model perplexity. Outlier splits (indicated with `*`) show high compression with minimal perplexity rise.
# 4. **Right Plot (KL Divergence vs. Effective Bits)**: Uses a logarithmic y-axis to represent the exact probability distribution distance (KL divergence) from the FP16 target, detailing how closely the soft probability space of token predictions is preserved.

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
for i, r in quality.iterrows():
    col = "C0" if r.config.startswith("TurboQuant") else ("C3" if r.config.startswith("INT") else "k")
    ax[0].scatter(r.bits_per_channel, r.perplexity, c=col, s=60)
    ax[0].annotate(r.config.replace(" (outlier split)", "*").replace(" + scale/zero per 32 ch", ""), (r.bits_per_channel, r.perplexity), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax[1].scatter(r.bits_per_channel, max(r.KL_vs_baseline, 1e-6), c=col, s=60)
ax[0].set(xlabel="effective bits per channel (incl. norms / scales)", ylabel="perplexity", title="Perplexity vs. memory (blue = TurboQuant, red = INT-b)")
ax[1].set(xlabel="effective bits per channel", ylabel="KL(FP16 || quantized)", yscale="log", title="Divergence from the FP16 model")
plt.tight_layout(); plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Sort results by agreement rate for a clean presentation
df_plot = quality.sort_values(by='top1_agree', ascending=False)

plt.figure(figsize=(10, 5))
colors = ['#1f77b4' if 'TurboQuant' in name else '#d62728' if 'INT' in name else '#7f7f7f' for name in df_plot['config']]

sns.barplot(
    x='top1_agree',
    y='config',
    data=df_plot,
    palette=colors,
    hue='config',
    legend=False
)

plt.title('Top-1 Token Agreement with FP16 Baseline')
plt.xlabel('Agreement Rate (Fraction of identical greedy tokens)')
plt.ylabel('Configuration')
plt.xlim(0, 1.05)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Annotate the exact percentages on the bars
for index, row in enumerate(df_plot.itertuples()):
    plt.text(row.top1_agree + 0.01, index, f"{row.top1_agree * 100:.1f}%", va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Explanation of the KV Cache Memory Footprint (Smaller is Better)
#
# This plot compares the physical memory required (`kv_MB`) to store the KV cache of 2048 tokens across configurations on Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128, so 128 KB per token in FP16):
#
# 1. **Uncompressed Cache (Baseline)**: **256 MB**.
# 2. **TurboQuant Savings**: **TurboQuant 4-bit** needs **66 MB** (3.9x smaller), **TurboQuant 2.5-bit** **40 MB** (6.4x) and **TurboQuant 2-bit** **34 MB** (7.5x).
# 3. **Metadata Overhead**: **INT2 per-group** quantization needs **48 MB**, more than TurboQuant 2.5-bit, because it stores a 16-bit scale and zero point for every group of 32 channels (1 extra bit per channel). TurboQuant's random rotation makes one fixed Lloyd-Max codebook fit every vector, so it stores only one fp16 norm per vector.
#
# These sizes are measured from the packed tensors and do not depend on the text.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Sort results by KV cache size in MB (smaller footprint is better)
df_mem = quality.sort_values(by='kv_MB', ascending=True)

plt.figure(figsize=(10, 5))
colors = ['#1f77b4' if 'TurboQuant' in name else '#d62728' if 'INT' in name else '#7f7f7f' for name in df_mem['config']]

sns.barplot(
    x='kv_MB',
    y='config',
    data=df_mem,
    palette=colors,
    hue='config',
    legend=False
)

plt.title('KV Cache Memory Footprint (Smaller is Better)')
plt.xlabel('KV Cache Footprint (MB) for 2048 Tokens')
plt.ylabel('Configuration')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Annotate the exact MB values on the bars
for index, row in enumerate(df_mem.itertuples()):
    plt.text(row.kv_MB + 0.5, index, f"{row.kv_MB:.2f} MB", va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Combined Analysis: Top-1 Agreement vs. Memory Footprint
#
# Below is a combined summary table showing both the accuracy preservation (Top-1 Token Agreement with the FP16 baseline) and the physical memory savings (`kv_MB`) for each KV-cache configuration. This lets you directly inspect which settings achieve the best compression with minimal quality degradation.

# %%
# Prepare a combined summary table
combined_df = quality[["config", "bits_per_channel", "compression", "top1_agree", "kv_MB"]].copy()
combined_df["top1_agree_pct"] = (combined_df["top1_agree"] * 100).round(1).astype(str) + "%"
combined_df = combined_df.rename(columns={
    "config": "Configuration",
    "bits_per_channel": "Bits/Channel",
    "compression": "Compression Ratio",
    "top1_agree_pct": "Top-1 Agreement",
    "kv_MB": "Memory (MB)"
}).drop(columns=["top1_agree"])

# Sort by memory footprint (smallest to largest)
combined_df = combined_df.sort_values(by="Memory (MB)", ascending=True).reset_index(drop=True)
display(combined_df)

# %% [markdown]
# ### Memory Footprint vs. Perplexity (Trade-off Curve)
#
# This line plot shows the trade-off between physical KV cache size (`kv_MB`) and perplexity.
# * The ideal configuration sits in the **bottom-left** (lowest memory, lowest perplexity).
# * Compare points at about the same memory: TurboQuant 3-bit (50 MB) against INT2 (48 MB), and TurboQuant 3.5-bit (60 MB) against INT3 (64 MB).

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Sort by memory footprint to make a sensible line/Pareto plot
df_line = quality.sort_values('kv_MB')

plt.figure(figsize=(10, 6))

# Plot the connected line to show the Pareto curve trend
plt.plot(df_line['kv_MB'], df_line['perplexity'], marker='', linestyle='-', color='gray', alpha=0.5)

# Overlay scatter points with distinct colors
for i, r in df_line.iterrows():
    col = "#1f77b4" if "TurboQuant" in r.config else ("#d62728" if "INT" in r.config else "k")
    plt.scatter(r.kv_MB, r.perplexity, c=col, s=80, zorder=5)

    # Add compact annotations for clarity
    label = r.config.replace(" (outlier split)", "*").replace(" + scale/zero per 32 ch", "")
    plt.annotate(
        label,
        (r.kv_MB, r.perplexity),
        fontsize=8,
        xytext=(6, -2),
        textcoords="offset points"
    )

plt.title("Memory Footprint vs. Perplexity (Trade-off Curve)")
plt.xlabel("KV Cache Footprint (MB) for 2048 Tokens")
plt.ylabel("Perplexity (Lower is Better)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Correlation Matrix: Quality vs. Memory Metrics
#
# This correlation matrix explores the linear relationships between the metrics collected during the quality evaluation. With only ten configurations it summarises the table above; it is not a statistical test.
# * **Bits & Perplexity**: You can observe the strong negative correlation between `bits_per_channel` and `perplexity` (fewer bits lead to higher/worse perplexity).
# * **Memory & Agreement**: A high positive correlation between `kv_MB` and `top1_agree` indicates that larger cache footprints preserve the model's exact greedy decisions.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Select numeric columns for the correlation matrix
numeric_cols = ['bits_per_channel', 'compression', 'perplexity', 'KL_vs_baseline', 'top1_agree', 'kv_MB']
corr_matrix = quality[numeric_cols].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    vmin=-1,
    vmax=1,
    linewidths=0.5
)

plt.title("Correlation Matrix of KV Cache Metrics")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Explanation of the Needle in a Haystack Task
#
# The "Needle in a Haystack" test is a standard long-context retrieval benchmark used to evaluate how well models can find and extract specific hidden statements (the "needle") within a massive context window of filler text (the "haystack").
#
# 1. **Setup**: We inject a secret statement (`NEEDLE = "The secret launch code..."`) at various depths ($0.1, 0.5, 0.9$) inside filler texts of lengths ranging from $2000$ to $8000$ tokens.
# 2. **Dynamic Generation**: The prompt is processed in a single prefill forward pass, and the KV cache is fully compressed. The model is then asked to output only the secret code.
# 3. **Evaluation**: A run counts as found only if the exact code appears in the answer. The haystack is the same WikiText-2 text used for the quality section, so the filler is real prose rather than a repeated sentence.

# %% [markdown]
# ## 4. Needle in a haystack
# We hide a fact at different depths of a long filler document, then ask for it. The whole prompt is prefilled once; every answer token is generated while attending to the **compressed** cache.

# %%
import importlib
import turboquant_core as tq
importlib.reload(tq)
from turboquant_core import (
    TurboQuant, MixedTurboQuant, UniformQuant, TurboQuantCache, fp16_cache_nbytes,
    tq_factory, mixed_factory, uniform_factory
)

# Re-bind the configurations to the newly reloaded factories
CONFIGS = {
    "Uncompressed cache (baseline)":            None,
    "TurboQuant 4-bit":                 (tq_factory(4), tq_factory(4)),
    "TurboQuant 3.5-bit (outlier split)": (mixed_factory(0.5, 4, 3), mixed_factory(0.5, 4, 3)),
    "TurboQuant 3-bit":                 (tq_factory(3), tq_factory(3)),
    "TurboQuant 2.5-bit (outlier split)": (mixed_factory(0.25, 3, 2), mixed_factory(0.25, 3, 2)),
    "TurboQuant 2-bit":                 (tq_factory(2), tq_factory(2)),
    "TurboQuant_prod 3-bit keys (MSE 2b + QJL 1b)": (tq_factory(3, "prod"), tq_factory(3)),
    "INT4 + scale/zero per 32 ch":      (uniform_factory(4), uniform_factory(4)),
    "INT3 + scale/zero per 32 ch":      (uniform_factory(3), uniform_factory(3)),
    "INT2 + scale/zero per 32 ch":      (uniform_factory(2), uniform_factory(2)),
}

if 'text' not in globals():
    text = ("The history of computing is a story of compression. " * 400)

NEEDLE = "The secret launch code for project Turbo is 7319-ALPHA-42."
QUESTION = "What is the secret launch code for project Turbo? Answer with the code only."
filler_src = text if len(text) > 50000 else (text * 20)

def build_prompt(n_tokens, depth):
    filler_ids = tok(filler_src, return_tensors="pt").input_ids[0, :n_tokens]
    cut = int(len(filler_ids) * depth)
    hay = tok.decode(filler_ids[:cut]) + " " + NEEDLE + " " + tok.decode(filler_ids[cut:])
    msgs = [{"role": "user", "content": f"Read the document and answer the question.\n\n<document>\n{hay}\n</document>\n\n{QUESTION}"}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(DEVICE)

@torch.no_grad()
def ask(prompt_ids, cache_name, max_new_tokens=16):
    cache = make_cache(cache_name, residual_length=128)
    out = model.generate(prompt_ids, past_key_values=cache, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)

LENGTHS = [2000, 4000, 8000] if DEVICE == "cuda" else [500, 1000]
DEPTHS = [0.1, 0.5, 0.9]
NIAH_CONFIGS = ["Uncompressed cache (baseline)", "TurboQuant 4-bit", "TurboQuant 3-bit", "TurboQuant 2.5-bit (outlier split)",
                "TurboQuant 2-bit", "INT2 + scale/zero per 32 ch"]
niah = []
for L in LENGTHS:
    for dep in DEPTHS:
        p = build_prompt(L, dep)
        for name in NIAH_CONFIGS:
            ans = ask(p, name)
            niah.append(dict(config=name, context=p.shape[1], depth=dep, found=int("7319-ALPHA-42" in ans), answer=ans.strip()[:40]))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
niah = pd.DataFrame(niah)
print(niah.pivot_table(index="config", columns=["context"], values="found", aggfunc="mean").reindex(NIAH_CONFIGS))

# %% [markdown]
# ### Sanity checks on the compressed cache
# The next three cells check the cache mechanics rather than the model: concatenating compressed blocks keeps every field aligned, the reconstruction error stays flat as tokens are appended one by one, and the raw answers of three configurations can be read directly.

# %%
# Let's run a precise diagnostic on Compressed.cat to see if shapes and scales align correctly during concatenation
import torch
from turboquant_core import Compressed, TurboQuant

d = 128
tq = TurboQuant(d, 4, device=DEVICE)

x1 = torch.randn(1, 2, 10, d, device=DEVICE)
x2 = torch.randn(1, 2, 1, d, device=DEVICE)

c1 = tq.quantize(x1)
c2 = tq.quantize(x2)

try:
    c_cat = c1.cat(c2, dim=2)
    print("Concatenation successful!")
    print("c1.idx shape:", c1.idx.shape)
    print("c2.idx shape:", c2.idx.shape)
    print("c_cat.idx shape:", c_cat.idx.shape)
    print("c1.norm shape:", c1.norm.shape)
    print("c2.norm shape:", c2.norm.shape)
    print("c_cat.norm shape:", c_cat.norm.shape)

    # Test dequantization accuracy after concatenation
    x_dequant = tq.dequantize(c_cat)
    x_orig = torch.cat([x1, x2], dim=2)
    error = (x_orig - x_dequant).abs().mean().item()
    print("Dequantization mean absolute error:", error)
except Exception as e:
    print("Error during concatenation/dequantization check:", e)

# %%
# Simulation of autoregressive generation using TurboQuantLayer to diagnose cumulative drift
import torch
import numpy as np
from turboquant_core import TurboQuantLayer, tq_factory

# Configure a 4-bit key/value quantizer layer
hd = 128
tq_layer = TurboQuantLayer(tq_factory(4)(0, hd), tq_factory(4)(0, hd), residual_length=128)

# Simulate a prefill of 256 tokens
prefill_k = torch.randn(1, 2, 256, hd, device=DEVICE)
prefill_v = torch.randn(1, 2, 256, hd, device=DEVICE)

tq_layer.lazy_initialization(prefill_k, prefill_v)
k_out, v_out = tq_layer.update(prefill_k, prefill_v)

# Simulate 5 steps of single-token autoregressive generation
steps_k = [torch.randn(1, 2, 1, hd, device=DEVICE) for _ in range(5)]
steps_v = [torch.randn(1, 2, 1, hd, device=DEVICE) for _ in range(5)]

curr_k = prefill_k
curr_v = prefill_v

for i in range(5):
    curr_k = torch.cat([curr_k, steps_k[i]], dim=2)
    curr_v = torch.cat([curr_v, steps_v[i]], dim=2)
    k_dec, v_dec = tq_layer.update(steps_k[i], steps_v[i])

    # Calculate error between fully reconstructed cache and step reference
    err_k = (curr_k - k_dec).abs().mean().item()
    err_v = (curr_v - v_dec).abs().mean().item()
    print(f"Step {i+1} - Cache length: {tq_layer.get_seq_length()} | Key MAE: {err_k:.6f} | Value MAE: {err_v:.6f}")

# %%
# Inspect the exact generated answers to check if the model is responding close to the needle
p_diag = build_prompt(2000, 0.5)
for name in ["Uncompressed cache (baseline)", "TurboQuant 4-bit", "TurboQuant 2.5-bit (outlier split)"]:
    ans = ask(p_diag, name)
    print(f"[{name}] generated answer:\n{repr(ans)}\n")

# %%
display(niah)

# %% [markdown]
# ### Visualizing Needle in a Haystack Retrieval Accuracy
#
# The heatmap below visualizes the retrieval accuracy of each quantization configuration across different context lengths (averaged across different depth placements of the needle).
#
# * **Values of 1.0 (Dark Blue)** indicate perfect retrieval.
# * **Lower values (Light Blue / Yellow)** indicate that the quantization noise caused the model to lose track of the specific long-range attention keys needed to find the hidden code.
# * Compare `INT2` (3.0 effective bits per channel, because of its scales) with `TurboQuant 2.5-bit` and `TurboQuant 2-bit`, which use fewer bits. Retrieval errors usually appear first at the longest context.
# * Nine prompts per configuration (three lengths times three depths) is a small sample: read a cell as "found in 3 of 3", "2 of 3", and so on.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Create a pivot table for the heatmap: config vs context length (averaged over depths)
heatmap_data = niah.pivot_table(index="config", columns="context", values="found", aggfunc="mean")

# Ensure the rows follow our defined logical order
heatmap_data = heatmap_data.reindex(NIAH_CONFIGS)

# Plot the heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    fmt=".2f",
    linewidths=.5,
    cbar_kws={'label': 'Retrieval Accuracy'}
)

plt.title("Needle in a Haystack: Retrieval Accuracy by Context Length")
plt.ylabel("KV Cache Configuration")
plt.xlabel("Context Length (Tokens)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Memory
# First the measured cache size for one long prompt, then the projection for a production-size Llama-3.1-8B model.

# %% [markdown]
# ### Detailed Explanation: Measuring Physical GPU Memory Footprint
#
# When deploying Large Language Models (LLMs) in production, raw theoretical bit-widths do not always translate to actual memory savings. The code block below directly measures the real physical GPU memory footprint of the Key-Value (KV) cache to demonstrate TurboQuant's memory-saving properties under a long-context scenario (using an 8080-token context prompt).
#
# #### Key Measurement Mechanisms:
# 1. **Dynamic Evaluation**: We process a long context prompt through the model to build up a substantial KV cache. For each configuration, we reset the PyTorch peak memory allocation tracker (`torch.cuda.reset_peak_memory_stats()`) and clean the garbage collector.
# 2. **KV Cache Size Calculation (`kv_cache_MB`)**: We calculate the exact size of the cache by analyzing the element size and tensor shapes inside our compressed containers. This corresponds to the physical bytes that must be moved between High-Bandwidth Memory (HBM) and the SRAM of the GPU stream multiprocessors during attention calculations.
# 3. **Overhead-Free Footprint**: Normal INT-based caches must store scaling factors and zero-point parameters per channel group. This metadata overhead creates a 'tax' that bloats the cache. By removing this requirement, TurboQuant approaches the theoretical lower bound of memory footprints at any target bit rate (e.g., 2-bit or 4-bit).

# %%
p = build_prompt(LENGTHS[-1], 0.5)
mem_rows = []
for name in ["Uncompressed cache (baseline)", "TurboQuant 4-bit", "TurboQuant 3.5-bit (outlier split)", "TurboQuant 3-bit",
             "TurboQuant 2.5-bit (outlier split)", "TurboQuant 2-bit", "INT4 + scale/zero per 32 ch", "INT2 + scale/zero per 32 ch"]:
    cache = make_cache(name)
    if DEVICE == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(p, past_key_values=cache, use_cache=True)
    mem_rows.append(dict(config=name, tokens=p.shape[1], kv_cache_MB=round(cache_nbytes(cache) / 2 ** 20, 2),
                         bits_per_channel=round(bits_per_channel(cache, p.shape[1]), 3),
                         peak_gpu_MB=round(torch.cuda.max_memory_allocated() / 2 ** 20) if DEVICE == "cuda" else None))
    del cache; gc.collect()
mem = pd.DataFrame(mem_rows)
mem["vs_baseline"] = (mem.kv_cache_MB[0] / mem.kv_cache_MB).round(2).astype(str) + "x smaller"
display(mem)

# %%
# What it means for Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128) on a 24 GB GPU after 16 GB of weights.
L_, H_, D_ = 32, 8, 128
def per_token_bytes(bits, norm_bits=16):
    return 2 * L_ * H_ * (bits * D_ + norm_bits) / 8
budget = 8 * 2 ** 30
proj = []
for label, bits, extra in [("FP16", 16, 0), ("INT4 + scale/zero (g=32)", 4, 32 * D_ // 32), ("TurboQuant 4-bit", 4, 16),
                           ("TurboQuant 3.5-bit", 3.5, 32), ("TurboQuant 2.5-bit", 2.5, 32)]:
    b = per_token_bytes(bits, extra)
    proj.append(dict(cache=label, KB_per_token=round(b / 1024, 1), max_context_in_8GB=int(budget / b)))
print(pd.DataFrame(proj).to_string(index=False))

# %% [markdown]
# ## 6. Speed (and why fused kernels matter)
# The reference cache dequantizes the whole history every step in PyTorch, so it **adds** latency. TurboQuant's speed claims (up to 8x faster attention logits at 4 bits on H100 in Google's blog) come from kernels that compute attention directly from the packed codes: the GPU reads 4x fewer bytes from HBM, and decode attention is memory-bound.

# %%
@torch.no_grad()
def decode_speed(name, prompt, n_new=64):
    cache = make_cache(name)
    model(prompt[:, :-1], past_key_values=cache, use_cache=True)
    nxt = prompt[:, -1:]
    if DEVICE == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(n_new):
        logits = model(nxt, past_key_values=cache, use_cache=True).logits
        nxt = logits[:, -1:].argmax(-1)
    if DEVICE == "cuda": torch.cuda.synchronize()
    return n_new / (time.time() - t)

p = build_prompt(LENGTHS[1], 0.5)
for name in ["Uncompressed cache (baseline)", "TurboQuant 4-bit", "TurboQuant 2.5-bit (outlier split)"]:
    print(f"{name:40s} {decode_speed(name, p):6.1f} tok/s  (context {p.shape[1]})")

# %% [markdown]
# ### Production path: vLLM
# vLLM (0.20.2+) ships fused TurboQuant kernels. On an Ampere/Hopper GPU:
# ```bash
# pip install -U vllm
# vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-cache-dtype turboquant_4bit_nc   # also: turboquant_k8v4, turboquant_k3v4_nc, turboquant_3bit_nc
# ```
# vLLM's blog reports 2.3-3.7x more KV-cache capacity, at 66-80% of BF16 throughput, with the aggressive 3-bit variants losing accuracy on hard reasoning tasks. FP8 remains the throughput-neutral option.
#
# ### Takeaways
# * The rotation makes every coordinate follow the same known distribution, so one precomputed codebook works for any data, with no calibration and no per-block scales.
# * At ~4 bits TurboQuant is close to lossless; 3.5 and 2.5 bits with outlier channels are the paper's sweet spots.
# * At the same total bits, uniform INT-b pays ~1 extra bit per channel for scales and zero points.
# * `TurboQuant_prod` is unbiased, but its QJL variance hurts softmax attention at low bits, which is why most implementations use the MSE variant (plus norm correction) for KV caches.
