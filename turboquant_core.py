"""
turboquant_core.py - a compact, pure-PyTorch reference implementation of TurboQuant.

Paper: Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate" (arXiv:2504.19874). Builds on QJL (arXiv:2406.03482) and
PolarQuant (arXiv:2502.02617).

What is implemented
  * Lloyd-Max codebooks for the exact coordinate distribution of a random unit vector
    (a scaled Beta distribution, Lemma 1 of the paper).
  * TurboQuant_mse  (Algorithm 1): random rotation -> per-coordinate optimal scalar quantizer.
  * TurboQuant_prod (Algorithm 2): TurboQuant_mse with b-1 bits + 1-bit QJL on the residual,
    giving an unbiased inner-product estimator.
  * Outlier-channel split ("2.5-bit" / "3.5-bit" configurations from Section 4.3).
  * Real bit-packing so memory numbers are measured, not estimated.
  * A uniform INT-b baseline with per-group fp16 scale + zero point (KIVI / HF QuantizedCache style).
  * A Hugging Face `transformers` (v5) KV-cache (`TurboQuantCache`) that stores keys/values
    compressed and dequantizes on read.

This is a readable reference, not a fused kernel: dequantization happens in PyTorch, so it
saves memory but does not speed attention up. Production speedups come from fused kernels
(e.g. vLLM `--kv-cache-dtype turboquant_*`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

# ----------------------------------------------------------------------------------------
# 1. Codebooks: Lloyd-Max for the coordinate distribution of a uniformly random unit vector
# ----------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def lloyd_max_codebook(d: int, bits: int, iters: int = 400, grid: int = 200_001) -> np.ndarray:
    """Optimal scalar quantizer (2**bits centroids) for f(x) ~ (1 - x^2)^((d-3)/2) on [-1, 1].

    This solves the continuous 1-D k-means problem of Eq. (4) numerically. Results are cached.
    """
    k = 2 ** bits
    x = np.linspace(-1.0, 1.0, grid)[1:-1]
    logf = 0.5 * (d - 3) * np.log1p(-x * x) if d > 3 else np.zeros_like(x)
    f = np.exp(logf - logf.max())
    f /= f.sum()
    # initialise at quantiles
    cdf = np.cumsum(f)
    c = np.interp((np.arange(k) + 0.5) / k, cdf, x)
    for _ in range(iters):
        b = 0.5 * (c[1:] + c[:-1])
        cell = np.searchsorted(b, x)
        w = np.bincount(cell, weights=f, minlength=k)
        m = np.bincount(cell, weights=f * x, minlength=k)
        c_new = np.where(w > 0, m / np.maximum(w, 1e-300), c)
        if np.max(np.abs(c_new - c)) < 1e-12:
            c = c_new
            break
        c = c_new
    return c.astype(np.float64)


def codebook_mse_cost(d: int, bits: int) -> float:
    """d * C(f_X, b): expected ||x - Q^-1(Q(x))||^2 for a unit vector (Theorem 1)."""
    c = lloyd_max_codebook(d, bits)
    x = np.linspace(-1.0, 1.0, 200_001)[1:-1]
    logf = 0.5 * (d - 3) * np.log1p(-x * x)
    f = np.exp(logf - logf.max()); f /= f.sum()
    b = 0.5 * (c[1:] + c[:-1])
    err = (x - c[np.searchsorted(b, x)]) ** 2
    return float(d * (err * f).sum())


# ----------------------------------------------------------------------------------------
# 2. Bit packing (any bit-width; exactly bits*d/8 bytes per vector)
# ----------------------------------------------------------------------------------------


def pack_bits(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer codes in [0, 2**bits) of shape [..., d] into uint8 of shape [..., bits*d/8]."""
    d = idx.shape[-1]
    assert d % 8 == 0, "last dim must be a multiple of 8"
    idx = idx.to(torch.uint8)
    shifts = torch.arange(bits, device=idx.device, dtype=torch.uint8)
    planes = (idx.unsqueeze(-2) >> shifts.view(-1, 1)) & 1            # [..., bits, d]
    planes = planes.reshape(*idx.shape[:-1], bits, d // 8, 8)
    weights = (1 << torch.arange(8, device=idx.device, dtype=torch.uint8))
    packed = (planes * weights).sum(-1, dtype=torch.uint8)             # [..., bits, d/8]
    return packed.reshape(*idx.shape[:-1], bits * d // 8)


_LUT = {}


def _bit_lut(device):
    if device not in _LUT:
        _LUT[device] = ((torch.arange(256, device=device).unsqueeze(1) >> torch.arange(8, device=device)) & 1).to(torch.uint8)
    return _LUT[device]


def unpack_bits(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Inverse of pack_bits -> int64 codes of shape [..., d] (one table lookup per byte)."""
    planes = _bit_lut(packed.device)[packed.long()]                      # [..., bits*d/8, 8]
    planes = planes.reshape(*packed.shape[:-1], bits, d)
    codes = planes[..., 0, :].clone()
    for j in range(1, bits):
        codes |= planes[..., j, :] << j
    return codes.long()


# ----------------------------------------------------------------------------------------
# 3. TurboQuant quantizers
# ----------------------------------------------------------------------------------------


def random_rotation(d: int, seed: int, device=None) -> torch.Tensor:
    """Haar-random orthogonal matrix via QR of a Gaussian matrix (Section 3.1)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(device=device, dtype=torch.float32)


@dataclass
class Compressed:
    """A batch of compressed vectors. Every field is a tensor with the same leading shape."""
    idx: torch.Tensor                    # packed MSE codes (uint8)
    norm: torch.Tensor                   # ||x|| in fp16
    qjl: torch.Tensor | None = None      # packed QJL sign bits (uint8), prod mode only
    gamma: torch.Tensor | None = None    # ||residual|| in fp16, prod mode only

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.idx, self.norm, self.qjl, self.gamma) if t is not None)

    def cat(self, other: "Compressed", dim: int) -> "Compressed":
        def c(a, b):
            return None if a is None else torch.cat([a, b], dim=dim)
        return Compressed(c(self.idx, other.idx), c(self.norm, other.norm), c(self.qjl, other.qjl), c(self.gamma, other.gamma))


class TurboQuant:
    """TurboQuant for d-dimensional vectors (arbitrary norm; the norm is stored in fp16).

    mode="mse"  -> Algorithm 1, `bits` bits per coordinate.
    mode="prod" -> Algorithm 2, (bits-1)-bit MSE stage + 1-bit QJL on the residual (unbiased <y, x>).
    renorm=True -> rescale the reconstruction so its norm equals the stored norm (cheap "norm
                   correction", similar in spirit to vLLM's *_nc variants).
    """

    def __init__(self, d: int, bits: int, mode: str = "mse", seed: int = 0, device=None, renorm: bool = False):
        assert mode in ("mse", "prod")
        assert d % 8 == 0
        self.d, self.bits, self.mode, self.renorm = d, bits, mode, renorm
        self.mse_bits = bits if mode == "mse" else bits - 1
        assert self.mse_bits >= 0
        self.device = device
        self.Pi = random_rotation(d, seed, device)                      # rotation (d x d)
        if self.mse_bits > 0:
            c = torch.tensor(lloyd_max_codebook(d, self.mse_bits), dtype=torch.float32, device=device)
            self.centroids = c
            self.boundaries = 0.5 * (c[1:] + c[:-1])
        if mode == "prod":
            g = torch.Generator().manual_seed(seed + 7919)
            self.S = torch.randn(d, d, generator=g).to(device)          # QJL projection
        self.bits_per_vector = bits * d + 16 + (16 if mode == "prod" else 0)

    def to(self, device):
        self.device = device
        for name in ("Pi", "centroids", "boundaries", "S"):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(device))
        return self

    # --- Algorithm 1 -------------------------------------------------------------------
    def _mse_codes(self, u: torch.Tensor) -> torch.Tensor:
        y = u @ self.Pi.T
        return torch.bucketize(y, self.boundaries)

    def _mse_decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.centroids[codes] @ self.Pi

    # --- public API -------------------------------------------------------------------
    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> Compressed:
        x = x.float()
        norm = x.norm(dim=-1, keepdim=True)
        u = x / norm.clamp_min(1e-12)
        if self.mse_bits > 0:
            codes = self._mse_codes(u)
            idx = pack_bits(codes, self.mse_bits)
            u_mse = self._mse_decode(codes)
        else:
            idx = torch.zeros(*x.shape[:-1], 0, dtype=torch.uint8, device=x.device)
            u_mse = torch.zeros_like(u)
        if self.mode == "mse":
            return Compressed(idx, norm.squeeze(-1).half())
        r = u - u_mse                                                    # residual
        signs = (r @ self.S.T) >= 0
        return Compressed(idx, norm.squeeze(-1).half(), pack_bits(signs, 1), r.norm(dim=-1).half())

    @torch.no_grad()
    def dequantize(self, c: Compressed, dtype=torch.float32) -> torch.Tensor:
        if self.mse_bits > 0:
            u = self._mse_decode(unpack_bits(c.idx, self.mse_bits, self.d))
        else:
            u = torch.zeros(*c.norm.shape, self.d, device=c.norm.device)
        if self.mode == "prod":
            z = unpack_bits(c.qjl, 1, self.d).float() * 2 - 1              # {-1,+1}
            u = u + (math.sqrt(math.pi / 2) / self.d) * c.gamma.float().unsqueeze(-1) * (z @ self.S)
        if self.renorm:
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return (u * c.norm.float().unsqueeze(-1)).to(dtype)

    def roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x), x.dtype)


class MixedTurboQuant:
    """Outlier split from Section 4.3: `n_out` outlier channels get `bits_hi`, the rest `bits_lo`.

    Example (paper): head_dim 128, 32 outliers @3 bits + 96 @2 bits = 2.5 bits/channel.
    Outlier channels are picked by `calibrate` (largest mean |x| on a sample, e.g. the prefill).
    """

    def __init__(self, d: int, n_out: int, bits_hi: int, bits_lo: int, mode="mse", seed=0, device=None, renorm=False):
        self.d, self.n_out = d, n_out
        self.hi = TurboQuant(n_out, bits_hi, mode, seed, device, renorm)
        self.lo = TurboQuant(d - n_out, bits_lo, mode, seed + 1, device, renorm)
        self.out_idx = torch.arange(n_out, device=device)
        self.in_idx = torch.arange(n_out, d, device=device)
        self.bits = (n_out * bits_hi + (d - n_out) * bits_lo) / d
        self.calibrated = False

    def calibrate(self, sample: torch.Tensor):
        score = sample.float().abs().reshape(-1, self.d).mean(0)
        order = torch.argsort(score, descending=True)
        self.out_idx, self.in_idx = order[: self.n_out].sort().values, order[self.n_out:].sort().values
        self.calibrated = True

    def to(self, device):
        self.hi.to(device); self.lo.to(device)
        self.out_idx, self.in_idx = self.out_idx.to(device), self.in_idx.to(device)
        return self

    def quantize(self, x):
        if not self.calibrated:
            self.calibrate(x)
        return (self.hi.quantize(x[..., self.out_idx]), self.lo.quantize(x[..., self.in_idx]))

    def dequantize(self, c, dtype=torch.float32):
        a, b = c
        lead = a.norm.shape
        out = torch.empty(*lead, self.d, device=a.norm.device, dtype=dtype)
        out[..., self.out_idx] = self.hi.dequantize(a, dtype)
        out[..., self.in_idx] = self.lo.dequantize(b, dtype)
        return out

    def roundtrip(self, x):
        return self.dequantize(self.quantize(x), x.dtype)


class UniformQuant:
    """Baseline: asymmetric min-max INT-b per group of `group` channels, fp16 scale + zero per group.

    Effective bits/channel = bits + 32/group (the "memory overhead" TurboQuant removes).
    """

    def __init__(self, d: int, bits: int, group: int = 32):
        assert d % group == 0
        self.d, self.bits, self.group = d, bits, group
        self.bits_per_vector = bits * d + 32 * (d // group)

    def to(self, device):
        return self

    @torch.no_grad()
    def quantize(self, x):
        g = x.float().reshape(*x.shape[:-1], self.d // self.group, self.group)
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        scale = ((hi - lo) / (2 ** self.bits - 1)).clamp_min(1e-8)
        q = torch.round((g - lo) / scale).clamp(0, 2 ** self.bits - 1)
        packed = pack_bits(q.reshape(*x.shape[:-1], self.d).long(), self.bits)
        return (packed, scale.half(), lo.half())

    @torch.no_grad()
    def dequantize(self, c, dtype=torch.float32):
        packed, scale, lo = c
        q = unpack_bits(packed, self.bits, self.d).float()
        q = q.reshape(*q.shape[:-1], self.d // self.group, self.group)
        return (q * scale.float() + lo.float()).reshape(*q.shape[:-2], self.d).to(dtype)

    def roundtrip(self, x):
        return self.dequantize(self.quantize(x), x.dtype)


def nbytes(c) -> int:
    """Bytes used by any compressed object returned by the quantizers above."""
    if isinstance(c, Compressed):
        return c.nbytes()
    if isinstance(c, (tuple, list)):
        return sum(nbytes(t) for t in c)
    if isinstance(c, torch.Tensor):
        return c.numel() * c.element_size()
    return 0


def cat_compressed(a, b, dim):
    if isinstance(a, Compressed):
        return a.cat(b, dim)
    if isinstance(a, (tuple, list)):
        return type(a)(cat_compressed(x, y, dim) for x, y in zip(a, b))
    return torch.cat([a, b], dim=dim)


# ----------------------------------------------------------------------------------------
# 4. Hugging Face transformers (v5) KV cache
# ----------------------------------------------------------------------------------------
try:
    from transformers.cache_utils import Cache, DynamicLayer

    class TurboQuantLayer(DynamicLayer):
        """One attention layer's cache: keys/values stored compressed, the last
        `residual_length` tokens kept in full precision (0 = quantize everything, as in the paper)."""

        def __init__(self, k_quant, v_quant, residual_length: int = 0):
            super().__init__()
            self.kq, self.vq = k_quant, v_quant
            self.residual_length = residual_length
            self.ck = self.cv = None          # compressed storage, token axis = -2 of the original
            self.cumulative_length = 0

        def lazy_initialization(self, key_states, value_states):
            super().lazy_initialization(key_states, value_states)
            for q in (self.kq, self.vq):
                q.to(key_states.device)
                if isinstance(q, MixedTurboQuant) and not q.calibrated:
                    q.calibrate(key_states if q is self.kq else value_states)

        def _append(self, k, v):
            ck, cv = self.kq.quantize(k), self.vq.quantize(v)
            if self.ck is None:
                self.ck, self.cv = ck, cv
            else:  # leading dims [B, H, T] -> token axis is 2 for every stored field
                self.ck, self.cv = cat_compressed(self.ck, ck, 2), cat_compressed(self.cv, cv, 2)

        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            self.cumulative_length += key_states.shape[-2]
            dt = key_states.dtype
            parts_k, parts_v = [], []
            if self.ck is not None:
                parts_k.append(self.kq.dequantize(self.ck, dt))
                parts_v.append(self.vq.dequantize(self.cv, dt))
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
            k_out = torch.cat(parts_k + [self.keys], dim=-2)
            v_out = torch.cat(parts_v + [self.values], dim=-2)
            overflow = self.keys.shape[-2] - self.residual_length
            if overflow > 0:  # move the oldest tokens from the fp16 buffer into compressed storage
                self._append(self.keys[..., :overflow, :], self.values[..., :overflow, :])
                self.keys = self.keys[..., overflow:, :].contiguous()
                self.values = self.values[..., overflow:, :].contiguous()
            return k_out, v_out

        def get_seq_length(self) -> int:
            return self.cumulative_length

        def reset(self):
            self.ck = self.cv = None
            self.cumulative_length = 0
            super().reset()

        def nbytes(self) -> int:
            fp = 0 if self.keys is None else self.keys.numel() * self.keys.element_size() * 2
            return fp + nbytes(self.ck) + nbytes(self.cv)

    class TurboQuantCache(Cache):
        """Drop-in KV cache for HF `generate`/forward.

        make_k / make_v: callables (layer_idx, head_dim) -> quantizer (TurboQuant, MixedTurboQuant,
        UniformQuant). Each layer gets its own random rotation via the seed.
        """

        def __init__(self, config, make_k, make_v, residual_length: int = 0):
            cfg = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
            hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
            layers = [TurboQuantLayer(make_k(i, hd), make_v(i, hd), residual_length) for i in range(cfg.num_hidden_layers)]
            super().__init__(layers=layers)

        def nbytes(self) -> int:
            return sum(l.nbytes() for l in self.layers)

    def fp16_cache_nbytes(cache) -> int:
        """Bytes of a regular DynamicCache."""
        return sum(l.keys.numel() * l.keys.element_size() + l.values.numel() * l.values.element_size()
                   for l in cache.layers if l.keys is not None)

except ImportError:  # transformers not installed: quantizers still work
    pass


# ----------------------------------------------------------------------------------------
# 5. Convenience factories
# ----------------------------------------------------------------------------------------

def tq_factory(bits, mode="mse", renorm=False, seed=0):
    def make(layer_idx, d):
        return TurboQuant(d, bits, mode=mode, seed=seed + 1000 * layer_idx, renorm=renorm)
    return make


def mixed_factory(n_out_frac, bits_hi, bits_lo, mode="mse", renorm=False, seed=0):
    def make(layer_idx, d):
        n_out = int(round(d * n_out_frac / 8)) * 8
        return MixedTurboQuant(d, n_out, bits_hi, bits_lo, mode=mode, seed=seed + 1000 * layer_idx, renorm=renorm)
    return make


def uniform_factory(bits, group=32):
    def make(layer_idx, d):
        return UniformQuant(d, bits, group)
    return make
