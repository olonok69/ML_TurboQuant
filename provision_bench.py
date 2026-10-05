"""
provision_bench.py - helpers for benchmarking vector compression on our own embeddings.

Used by provision_search_benchmark.ipynb (which embeds this file with %%writefile).
Pure numpy + torch; turbovec is optional. Nothing here talks to Qdrant.

Conventions
  * X: float32 [n, d] database vectors, already normalized when the metric is cosine.
  * Queries are database points ("find provisions similar to this provision"), given as
    row indices `qidx`; each query's own row is excluded from its results.
  * Scores are inner products (cosine for normalized vectors). Higher = more similar.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from turboquant_core import TurboQuant, unpack_bits

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------------------
# exact search and metrics
# ----------------------------------------------------------------------------------------


def normalize(X: np.ndarray) -> np.ndarray:
    return (X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def _topk_excluding_self(score: torch.Tensor, qidx_chunk: torch.Tensor, offset: int, k: int):
    """Mask each query's own row (if it falls inside this database chunk) and take top-k."""
    rows = torch.arange(score.shape[0], device=score.device)
    local = qidx_chunk - offset
    inside = (local >= 0) & (local < score.shape[1])
    score[rows[inside], local[inside]] = -float("inf")
    kk = min(k, score.shape[1])
    s, i = score.topk(kk, dim=1)
    return s, i + offset


def merge_topk(best_s, best_i, s, i, k):
    cs, ci = torch.cat([best_s, s], 1), torch.cat([best_i, i], 1)
    s2, top = cs.topk(k, dim=1)
    return s2, ci.gather(1, top)


@torch.no_grad()
def exact_topk(X: np.ndarray, qidx: np.ndarray, k: int, chunk: int = 50_000, qbatch: int = 1024):
    """Exact inner-product top-k for the query rows, self excluded. Returns (scores, indices) numpy."""
    Xt = torch.from_numpy(X).to(DEVICE)
    out_s, out_i = [], []
    for a in range(0, len(qidx), qbatch):
        qi = torch.from_numpy(qidx[a:a + qbatch]).to(DEVICE)
        q = Xt[qi]
        best_s = torch.full((len(qi), k), -float("inf"), device=DEVICE)
        best_i = torch.full((len(qi), k), -1, dtype=torch.long, device=DEVICE)
        for s0 in range(0, Xt.shape[0], chunk):
            sc = q @ Xt[s0:s0 + chunk].T
            s, i = _topk_excluding_self(sc, qi, s0, k)
            best_s, best_i = merge_topk(best_s, best_i, s, i, k)
        out_s.append(best_s.cpu()); out_i.append(best_i.cpu())
    return torch.cat(out_s).numpy(), torch.cat(out_i).numpy()


def recall_at_k(I_approx: np.ndarray, I_true: np.ndarray, k: int) -> float:
    """Average overlap between the true top-k and the returned top-k (k@k)."""
    return float(np.mean([len(set(a[:k]) & set(t[:k])) / k for a, t in zip(I_approx, I_true)]))


def r1_at_k(I_approx: np.ndarray, I_true: np.ndarray, k: int) -> float:
    """How often the true nearest neighbour is inside the returned top-k (the paper's Recall@1@k)."""
    return float(np.mean([t[0] in set(a[:k]) for a, t in zip(I_approx, I_true)]))


# ----------------------------------------------------------------------------------------
# compression methods (each mimics a Qdrant quantization option, scored asymmetrically:
# the query stays in float32, only stored vectors are compressed, as Qdrant does)
# ----------------------------------------------------------------------------------------


class Method:
    name = "base"
    bits = 32.0
    reconstructs = False

    def build(self, X: np.ndarray):
        raise NotImplementedError

    def scores(self, q: torch.Tensor, s0: int, s1: int) -> torch.Tensor:
        """Approximate scores of float queries q [m, d] against stored rows s0:s1."""
        raise NotImplementedError

    def reconstruct(self, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def nbytes(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def search(self, X: np.ndarray, qidx: np.ndarray, k: int, chunk: int = 50_000, qbatch: int = 1024):
        Xt = torch.from_numpy(X).to(DEVICE)
        out_s, out_i = [], []
        for a in range(0, len(qidx), qbatch):
            qi = torch.from_numpy(qidx[a:a + qbatch]).to(DEVICE)
            q = Xt[qi]
            best_s = torch.full((len(qi), k), -float("inf"), device=DEVICE)
            best_i = torch.full((len(qi), k), -1, dtype=torch.long, device=DEVICE)
            for s0 in range(0, self.n, chunk):
                sc = self.scores(q, s0, min(s0 + chunk, self.n))
                s, i = _topk_excluding_self(sc, qi, s0, k)
                best_s, best_i = merge_topk(best_s, best_i, s, i, k)
            out_s.append(best_s.cpu()); out_i.append(best_i.cpu())
        return torch.cat(out_s).numpy(), torch.cat(out_i).numpy()


class Float32(Method):
    name, bits, reconstructs = "float32 (no quantization)", 32.0, True

    def build(self, X):
        self.X = torch.from_numpy(X).to(DEVICE); self.n = len(X); return self

    def scores(self, q, s0, s1):
        return q @ self.X[s0:s1].T

    def reconstruct(self, idx):
        return self.X[torch.from_numpy(idx).to(DEVICE)].cpu().numpy()

    def nbytes(self):
        return self.X.numel() * 4


class ScalarInt8(Method):
    """Like Qdrant scalar quantization (int8): one global range from the `quantile` of all values."""
    bits, reconstructs = 8.0, True

    def __init__(self, quantile: float = 0.99):
        self.quantile = quantile
        self.name = f"scalar int8 (quantile {quantile})"

    def build(self, X):
        flat = X.reshape(-1)
        sample = flat[np.random.default_rng(0).choice(flat.size, min(flat.size, 2_000_000), replace=False)]
        lo, hi = np.quantile(sample, [(1 - self.quantile) / 2, 1 - (1 - self.quantile) / 2])
        self.lo, self.step = float(lo), float(hi - lo) / 255
        codes = np.clip(np.round((X - self.lo) / self.step), 0, 255).astype(np.uint8)
        self.codes = torch.from_numpy(codes).to(DEVICE); self.n = len(X)
        return self

    def _deq(self, s0, s1):
        return self.codes[s0:s1].float() * self.step + self.lo

    def scores(self, q, s0, s1):
        return q @ self._deq(s0, s1).T

    def reconstruct(self, idx):
        return (self.codes[torch.from_numpy(idx).to(DEVICE)].float() * self.step + self.lo).cpu().numpy()

    def nbytes(self):
        return self.codes.numel()


class Binary1(Method):
    """Like Qdrant binary quantization: one sign bit per dimension, query kept in float."""
    name, bits, reconstructs = "binary 1-bit", 1.0, False

    def build(self, X):
        self.packed = torch.from_numpy(np.packbits(X > 0, axis=1)).to(DEVICE)
        self.d, self.n = X.shape[1], len(X)
        return self

    def scores(self, q, s0, s1):
        bits = torch.from_numpy(np.unpackbits(self.packed[s0:s1].cpu().numpy(), axis=1)[:, :self.d]).to(DEVICE)
        return q @ (bits.float() * 2 - 1).T

    def nbytes(self):
        return self.packed.numel()


class TurboQuantRef(Method):
    """TurboQuant_mse from turboquant_core (random rotation + Lloyd-Max codebook), optional norm correction.

    renorm=True rescales each reconstructed vector to its stored norm, similar in spirit to the length
    renormalization Qdrant adds in its own TurboQuant implementation.
    """
    reconstructs = True

    def __init__(self, bits: int, renorm: bool = True, seed: int = 0):
        self.bits, self.renorm, self.seed = float(bits), renorm, seed
        self.name = f"TurboQuant {bits}-bit" + (" + norm correction" if renorm else "")

    def build(self, X, batch: int = 50_000):
        d = X.shape[1]
        assert d % 8 == 0, "TurboQuant reference needs a dimension divisible by 8"
        self.q = TurboQuant(d, int(self.bits), "mse", seed=self.seed, device=DEVICE, renorm=self.renorm)
        parts = [self.q.quantize(torch.from_numpy(X[i:i + batch]).to(DEVICE)) for i in range(0, len(X), batch)]
        self.idx = torch.cat([p.idx for p in parts]); self.norm = torch.cat([p.norm for p in parts])
        self.d, self.n = d, len(X)
        # renorm factor = ||x|| / ||c[idx]|| (the rotation keeps lengths, so it is computed in the rotated space)
        if self.renorm:
            cn = []
            for s0 in range(0, self.n, batch):
                y = self.q.centroids[unpack_bits(self.idx[s0:s0 + batch], int(self.bits), d)]
                cn.append(y.norm(dim=1))
            self.scale = self.norm.float() / torch.cat(cn).clamp_min(1e-12)
        else:
            self.scale = self.norm.float()
        return self

    def scores(self, q, s0, s1):
        y = self.q.centroids[unpack_bits(self.idx[s0:s1], int(self.bits), self.d)]   # rotated reconstruction
        return ((q @ self.q.Pi.T) @ y.T) * self.scale[s0:s1]

    def reconstruct(self, idx):
        ii = torch.from_numpy(idx).to(DEVICE)
        y = self.q.centroids[unpack_bits(self.idx[ii], int(self.bits), self.d)]
        return ((y @ self.q.Pi) * self.scale[ii].unsqueeze(1)).cpu().numpy()

    def nbytes(self):
        return self.idx.numel() + self.norm.numel() * 2


class Turbovec(Method):
    """turbovec (Rust + SIMD TurboQuant). Optional: only if `pip install turbovec` worked."""
    reconstructs = False

    def __init__(self, bits: int):
        self.bits = float(bits); self.name = f"turbovec {bits}-bit (SIMD)"

    def build(self, X):
        import turbovec
        self.ix = turbovec.TurboQuantIndex(dim=X.shape[1], bit_width=int(self.bits))
        self.ix.add(X); self.ix.prepare(); self.n = len(X)
        return self

    def search(self, X, qidx, k, **_):
        s, i = self.ix.search(np.ascontiguousarray(X[qidx]), k + 1)
        s, i = np.asarray(s), np.asarray(i).astype(np.int64)
        return drop_self(s, i, qidx, k)

    def nbytes(self):
        return len(self.ix.to_bytes())


def drop_self(S: np.ndarray, I: np.ndarray, qidx: np.ndarray, k: int):
    """Remove each query's own index from a (k+1)-wide result and keep k columns."""
    keep = I != qidx[:, None]
    S2 = np.stack([s[m][:k] for s, m in zip(S, keep)]); I2 = np.stack([i[m][:k] for i, m in zip(I, keep)])
    return S2, I2


def rescore(X: np.ndarray, qidx: np.ndarray, I_cand: np.ndarray, k: int):
    """Re-rank candidate lists with exact float32 scores (what Qdrant's `rescore=True` does)."""
    q = X[qidx]
    S = np.einsum("qd,qcd->qc", q, X[I_cand])
    order = np.argsort(-S, axis=1)[:, :k]
    return np.take_along_axis(S, order, 1), np.take_along_axis(I_cand, order, 1)


# ----------------------------------------------------------------------------------------
# benchmark driver
# ----------------------------------------------------------------------------------------


def run_method(method: Method, X, qidx, I_true, k=10, oversampling=(1, 2, 4), raw_bytes=None):
    """Build once, then search without and with rescoring at each oversampling factor."""
    raw_bytes = raw_bytes or X.nbytes
    t = time.time(); method.build(X)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    build_s = time.time() - t
    rows = []
    for ov in oversampling:
        if isinstance(method, Float32) and ov > 1:
            continue  # exact scores already; rescoring changes nothing
        t = time.time(); S, I = method.search(X, qidx, k * ov)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        search_s = time.time() - t
        if ov == 1:
            variants = [("no rescore", I[:, :k], search_s)]
        else:
            t = time.time(); _, I2 = rescore(X, qidx, I, k)
            variants = [(f"rescore, oversampling {ov}x", I2, search_s + time.time() - t)]
        for label, Ik, sec in variants:
            rows.append(dict(method=method.name, search=label, bits_per_dim=method.bits,
                             MB=round(method.nbytes() / 2 ** 20, 1), compression=round(raw_bytes / method.nbytes(), 1),
                             build_s=round(build_s, 2), QPS=round(len(qidx) / max(sec, 1e-9)),
                             **{f"recall@{k}": round(recall_at_k(Ik, I_true, k), 4),
                                "top1_found_in_top1": round(r1_at_k(Ik, I_true, 1), 4),
                                f"top1_found_in_top{k}": round(r1_at_k(Ik, I_true, k), 4)}))
    return rows


# ----------------------------------------------------------------------------------------
# embedding health (quality diagnostics that need no labels)
# ----------------------------------------------------------------------------------------


def embedding_health(X_raw: np.ndarray, X: np.ndarray, S_true: np.ndarray, I_true: np.ndarray, n_pairs: int = 20_000):
    g = np.random.default_rng(0)
    norms = np.linalg.norm(X_raw, axis=1)
    a, b = g.integers(0, len(X), n_pairs), g.integers(0, len(X), n_pairs)
    rand_cos = np.einsum("nd,nd->n", X[a], X[b])
    mean_dir = float(np.linalg.norm(X.mean(0)))
    # effective dimension: components needed for 90% of the variance (on a sample)
    samp = X[g.choice(len(X), min(len(X), 20_000), replace=False)]
    ev = np.linalg.svd(samp - samp.mean(0), compute_uv=False) ** 2
    eff90 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.90) + 1)
    # duplicates and hubs
    rounded = np.round(X, 5)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    exact_dups = int((counts - 1).clip(min=0).sum())
    occ = np.bincount(I_true[:, :10].reshape(-1), minlength=len(X))
    hub_share = float(np.sort(occ)[::-1][: max(1, len(X) // 100)].sum() / max(occ.sum(), 1))
    return dict(
        vectors=len(X), dim=X.shape[1],
        norm_min=float(norms.min()), norm_median=float(np.median(norms)), norm_max=float(norms.max()),
        already_normalized=bool(np.allclose(norms, 1, atol=1e-3)),
        random_pair_cos_mean=float(rand_cos.mean()), random_pair_cos_p95=float(np.quantile(rand_cos, 0.95)),
        mean_direction_norm=mean_dir, effective_dim_90pct=eff90,
        exact_duplicate_vectors=exact_dups,
        nn1_score_median=float(np.median(S_true[:, 0])),
        near_duplicate_queries_pct=float((S_true[:, 0] > 0.99).mean() * 100),
        top1pct_hubs_share_of_top10_slots=hub_share,
    ), rand_cos, occ


# ----------------------------------------------------------------------------------------
# auto-merge analysis
# ----------------------------------------------------------------------------------------


def candidate_pairs(qidx: np.ndarray, S_true: np.ndarray, I_true: np.ndarray, min_score: float):
    """Pairs (a, b, exact score) among each query's exact top-k with score >= min_score, deduplicated."""
    seen, rows = set(), []
    for q, srow, irow in zip(qidx, S_true, I_true):
        for s, j in zip(srow, irow):
            if s < min_score:
                break
            key = (min(q, j), max(q, j))
            if key not in seen:
                seen.add(key); rows.append((key[0], key[1], float(s)))
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    a, b, s = map(np.array, zip(*rows))
    return a.astype(np.int64), b.astype(np.int64), s.astype(np.float32)


def approx_pair_scores(method: Method, X: np.ndarray, a: np.ndarray, b: np.ndarray, batch: int = 50_000):
    """Score of each pair as a quantized search would see it: float query a against stored (compressed) b."""
    out = []
    for s0 in range(0, len(a), batch):
        rb = method.reconstruct(b[s0:s0 + batch])
        out.append(np.einsum("nd,nd->n", X[a[s0:s0 + batch]], rb))
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def merge_flips(exact: np.ndarray, approx: np.ndarray, threshold: float) -> dict:
    e, p = exact >= threshold, approx >= threshold
    return dict(pairs_checked=int(len(exact)), merges_exact=int(e.sum()),
                false_merges=int((p & ~e).sum()), missed_merges=int((e & ~p).sum()),
                score_error_mean=float(np.mean(approx - exact)) if len(exact) else 0.0,
                score_error_p95_abs=float(np.quantile(np.abs(approx - exact), 0.95)) if len(exact) else 0.0)


# ----------------------------------------------------------------------------------------
# labelled evaluation (optional CSVs from domain experts)
# ----------------------------------------------------------------------------------------


def recall_from_labels(I: np.ndarray, qidx: np.ndarray, relevant: dict, k: int = 10) -> float:
    """relevant: {query row index -> set of relevant row indices}. Average share of them found in the top-k."""
    vals = []
    for q, row in zip(qidx, I):
        rel = relevant.get(int(q))
        if rel:
            vals.append(len(set(row[:k]) & rel) / len(rel))
    return float(np.mean(vals)) if vals else float("nan")


def merge_pr_curve(scores: np.ndarray, same: np.ndarray, thresholds: np.ndarray):
    """Precision and recall of 'merge if score >= t' against labelled pairs (same = 1 for true duplicates)."""
    rows = []
    for t in thresholds:
        pred = scores >= t
        tp = int((pred & (same == 1)).sum()); fp = int((pred & (same == 0)).sum()); fn = int((~pred & (same == 1)).sum())
        rows.append(dict(threshold=float(t), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                         merges=int(pred.sum()), wrong_merges=fp))
    return rows
