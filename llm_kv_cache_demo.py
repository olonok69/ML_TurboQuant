# %% [markdown]
# # TurboQuant KV-cache compression: LLM inference demo (Google Colab)
#
# This notebook reproduces the core ideas of **TurboQuant** (Zandieh, Daliri, Hadian, Mirrokni, Google Research, arXiv:2504.19874)
# on a small open model, end to end, in pure PyTorch:
#
# 1. **Theory check.** TurboQuant's measured distortion versus the paper's upper and lower bounds, and the bias of MSE quantizers versus the unbiased `TurboQuant_prod` (MSE + 1-bit QJL).
# 2. **Quality.** Perplexity, KL divergence and top-1 agreement versus the FP16 cache at 4, 3.5, 3, 2.5 and 2 bits, against a classic INT-b baseline that has to store per-group scales and zero points.
# 3. **Needle in a haystack.** Long-context retrieval with the compressed cache.
# 4. **Memory.** Measured KV-cache bytes and GPU memory, plus what it means for an 8B model.
# 5. **Speed.** An honest look at what a reference implementation costs, and where the real speedups come from (fused kernels, e.g. vLLM).
#
# **Runtime:** `Runtime > Change runtime type > T4 GPU` (free tier is enough). Everything also runs on CPU with `MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"`, just slower.
#
# > The reference implementation stores keys and values bit-packed and dequantizes them on read. That saves memory but adds work, so it is **slower** than the FP16 cache. Production speedups need fused kernels (vLLM ships them: `--kv-cache-dtype turboquant_4bit_nc`).
#
# **Companion reading:** the workshop guide (`docs/guide/TurboQuant_Technical_Guide_EN.md` in the repository; Spanish version `TurboQuant_Guia_Tecnica_ES.md`). Section 3 explains the algorithm, section 4 the KV-cache use case, and section 6.1 walks through this notebook section by section.

# %%
# !pip -q install -U "transformers>=5.0,<6" accelerate datasets matplotlib pandas
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
for p, s in [("datasets", "datasets"), ("matplotlib", "matplotlib"), ("pandas", "pandas")]:
    _ensure(p, s)
print("transformers", transformers.__version__)

# %% [markdown]
# ## The TurboQuant implementation
# The next cell writes `turboquant_core.py` (about 400 lines, readable). Key pieces:
# * `lloyd_max_codebook(d, b)`: optimal scalar quantizer for the coordinate distribution of a random unit vector (a Beta distribution, close to N(0, 1/d)).
# * `TurboQuant(d, bits, mode="mse")`: random rotation, then nearest centroid per coordinate (Algorithm 1). Stores `bits*d` bits plus one fp16 norm.
# * `TurboQuant(d, bits, mode="prod")`: (b-1)-bit MSE stage plus a 1-bit QJL sketch of the residual (Algorithm 2). Unbiased inner products.
# * `MixedTurboQuant`: outlier-channel split used for the paper's 2.5-bit and 3.5-bit settings.
# * `UniformQuant`: baseline INT-b with fp16 scale and zero point per group of 32 channels (+1 bit per channel of overhead).
# * `TurboQuantCache`: a Hugging Face `Cache` that keeps K/V compressed.

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
# `Qwen2.5-1.5B-Instruct` (28 layers, 2 KV heads, head_dim 128) fits a free T4 in FP16.
# Head dim 128 lets us use exactly the paper's outlier split: 32 channels at 3 bits + 96 at 2 bits, which the paper calls **2.5 bits**. (The codes alone are 2.25 bits/channel; with the two fp16 norms the measured cost is exactly 2.5. Likewise the 3.5-bit split, 64 @4b + 64 @3b, measures 3.75 with norms. The `bits_per_channel` column below is always measured from the packed tensors.)

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE == "cuda" else "Qwen/Qwen2.5-0.5B-Instruct"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
cfg = model.config
HEAD_DIM = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
print(MODEL_ID, "| layers", cfg.num_hidden_layers, "| kv heads", cfg.num_key_value_heads, "| head_dim", HEAD_DIM)

# Configurations to compare. Each value: (key quantizer factory, value quantizer factory) or None for FP16.
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
# Metrics against the FP16 run: perplexity, mean KL(FP16 ‖ quantized) of the next-token distribution, and top-1 agreement.

# %%
from datasets import load_dataset
try:
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
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
results = []
for name in CONFIGS:
    t0 = time.time()
    logits, cache = (ref_logits, ref_cache) if CONFIGS[name] is None else chunked_logits(name)
    logp = torch.log_softmax(logits[0, :-1], -1)
    nll = -logp.gather(-1, targets[:, None]).mean().item()
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1).mean().item()
    agree = (logp.argmax(-1) == ref_logp.argmax(-1)).float().mean().item()
    results.append(dict(config=name, bits_per_channel=round(bits_per_channel(cache, ids.shape[1]), 3),
                        compression=round(16 / bits_per_channel(cache, ids.shape[1]), 2),
                        perplexity=round(math.exp(nll), 3), KL_vs_baseline=round(kl, 5), top1_agree=round(agree, 4),
                        kv_MB=round(cache_nbytes(cache) / 2 ** 20, 2)))
    del cache; gc.collect()
quality = pd.DataFrame(results)
display(quality)

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

# %% [markdown]
# ## 4. Needle in a haystack
# We hide a fact at different depths of a long filler document, then ask for it. The whole prompt is prefilled once; every answer token is generated while attending to the **compressed** cache.

# %%
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
    cache = make_cache(cache_name)
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
# ## 5. Memory
# First the measured cache size for one long prompt, then the projection for a production-size model.

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
# vLLM (0.20.2+) ships fused TurboQuant kernels. On an Ampere/Hopper GPU (not the free T4):
# ```bash
# pip install -U vllm
# vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype turboquant_4bit_nc   # also: turboquant_k8v4, turboquant_k3v4_nc, turboquant_3bit_nc
# ```
# vLLM's blog reports 2.3-3.7x more KV-cache capacity, at 66-80% of BF16 throughput, with the aggressive 3-bit variants losing accuracy on hard reasoning tasks. FP8 remains the throughput-neutral option.
#
# ### Takeaways
# * The rotation makes every coordinate follow the same known distribution, so one precomputed codebook works for any data, with no calibration and no per-block scales.
# * At ~4 bits TurboQuant is close to lossless; 3.5 and 2.5 bits with outlier channels are the paper's sweet spots.
# * At the same total bits, uniform INT-b pays ~1 extra bit per channel for scales and zero points.
# * `TurboQuant_prod` is unbiased, but its QJL variance hurts softmax attention at low bits, which is why most implementations use the MSE variant (plus norm correction) for KV caches.
