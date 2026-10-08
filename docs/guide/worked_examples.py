"""Reproduce the small-number worked examples in sections 2.1 to 2.3 of the guides.

All three examples use the same 4-dimensional vector v = [3, -4, 2, 0.5] and the same
query q = [1, -1, 1, 0], whose exact inner product is 9. Run with plain NumPy:

    python docs/guide/worked_examples.py

The Lloyd-Max solver below is the one in turboquant_core.py, copied so this script does
not need PyTorch.
"""

import numpy as np

V = np.array([3.0, -4.0, 2.0, 0.5])
Q = np.array([1.0, -1.0, 1.0, 0.0])
D = 4

# Gaussian matrix of the QJL example (3 x 4); TurboQuant adds a fourth row (d x d).
S_QJL = np.array([[0.5, -0.2, 0.8, -0.1],
                  [-0.9, 0.1, 0.3, 0.7],
                  [0.2, 0.6, -0.4, -0.5]])
S_TQ = np.vstack([S_QJL, [0.4, -0.7, -0.3, 0.9]])

# Rotation of the TurboQuant example: a Hadamard matrix with random column signs, scaled
# by 1/2 so it is orthogonal. A fast random rotation with entries that are easy to multiply.
H = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float)
SIGNS = np.array([1.0, -1.0, 1.0, -1.0])
PI = 0.5 * H * SIGNS


def lloyd_max_codebook(d, bits, iters=400, grid=200_001):
    """Optimal scalar quantizer for f(x) ~ (1 - x^2)^((d-3)/2) on [-1, 1] (as in turboquant_core.py)."""
    k = 2 ** bits
    x = np.linspace(-1.0, 1.0, grid)[1:-1]
    logf = 0.5 * (d - 3) * np.log1p(-x * x) if d > 3 else np.zeros_like(x)
    f = np.exp(logf - logf.max())
    f /= f.sum()
    c = np.interp((np.arange(k) + 0.5) / k, np.cumsum(f), x)
    for _ in range(iters):
        b = 0.5 * (c[1:] + c[:-1])
        cell = np.searchsorted(b, x)
        w = np.bincount(cell, weights=f, minlength=k)
        m = np.bincount(cell, weights=f * x, minlength=k)
        c_new = np.where(w > 0, m / np.maximum(w, 1e-300), c)
        if np.max(np.abs(c_new - c)) < 1e-12:
            return c_new
        c = c_new
    return c


def haar_rotation(d, rng):
    a = rng.standard_normal((d, d))
    qm, r = np.linalg.qr(a)
    return qm * np.sign(np.diag(r))


def qjl_correction(s, q, r):
    """sqrt(pi/2)/m * ||r|| * <S q, sign(S r)>, the QJL estimate of <q, r>."""
    m = s.shape[0]
    return np.sqrt(np.pi / 2) / m * np.linalg.norm(r) * ((s @ q) @ np.sign(s @ r))


def qjl():
    print("== 2.1 QJL (m = 3) ==")
    norm = np.linalg.norm(V)
    sk, sq = S_QJL @ V, S_QJL @ Q
    est = np.sqrt(np.pi / 2) / 3 * norm * (sq @ np.sign(sk))
    print(f"||k|| = {norm:.4f}  S.k = {sk}  signs = {np.sign(sk)}")
    print(f"S.q = {sq}  <S.q, sign> = {sq @ np.sign(sk):.2f}  estimate = {est:.2f}  (exact {Q @ V:.0f})")


def polarquant():
    print("== 2.2 PolarQuant (d = 4, two levels) ==")
    ra, rb = np.hypot(V[0], V[1]), np.hypot(V[2], V[3])
    t1 = np.degrees(np.arctan2(V[1], V[0])) % 360
    t2 = np.degrees(np.arctan2(V[3], V[2])) % 360
    psi = np.degrees(np.arctan2(rb, ra))
    # level-2 codebook: 1-D k-means (Lloyd) on the density sin(2 psi) over [0, 90 deg]
    x = np.linspace(0, 90, 200_001)[1:-1]
    f = np.sin(np.radians(2 * x))
    c = np.array([11.25, 33.75, 56.25, 78.75])
    for _ in range(500):
        cell = np.searchsorted(0.5 * (c[1:] + c[:-1]), x)
        c = np.bincount(cell, f * x, 4) / np.bincount(cell, f, 4)
    t1q = (np.floor(t1 / 22.5) + 0.5) * 22.5
    t2q = (np.floor(t2 / 22.5) + 0.5) * 22.5
    psiq = c[np.argmin(np.abs(c - psi))]
    norm = np.linalg.norm(V)
    ra_h, rb_h = norm * np.cos(np.radians(psiq)), norm * np.sin(np.radians(psiq))
    v_hat = np.array([ra_h * np.cos(np.radians(t1q)), ra_h * np.sin(np.radians(t1q)),
                      rb_h * np.cos(np.radians(t2q)), rb_h * np.sin(np.radians(t2q))])
    print(f"R_A = {ra:.2f}  R_B = {rb:.2f}  theta1 = {t1:.1f}  theta2 = {t2:.1f}  psi = {psi:.1f}")
    print(f"level-2 centroids = {np.round(c, 1)}  quantized: {t1q}, {t2q}, {psiq:.1f}")
    print(f"v_hat = {np.round(v_hat, 2)}  rel. error = {np.linalg.norm(V - v_hat) / norm:.3f}"
          f"  <q, v_hat> = {Q @ v_hat:.2f}")


def turboquant():
    print("== 2.3 TurboQuant_prod, b = 3 (2-bit MSE stage + 1-bit QJL) ==")
    norm = np.linalg.norm(V)
    c = lloyd_max_codebook(D, 2)
    bounds = 0.5 * (c[1:] + c[:-1])
    print(f"1. ||v|| = {norm:.4f}  u = v/||v|| = {np.round(V / norm, 4)}")
    y_raw = PI @ V
    y = y_raw / norm
    print(f"2. Pi.v = {y_raw}  (||Pi.v||^2 = {y_raw @ y_raw:.2f})  y = Pi.v/||v|| = {np.round(y, 4)}")
    idx = np.searchsorted(bounds, y)
    print(f"3. codebook d=4, 2 bits = {np.round(c, 4)}  boundaries = {np.round(bounds, 4)}")
    print(f"   idx = {idx}  y_hat = {np.round(c[idx], 4)}")
    x_mse = norm * (PI.T @ c[idx])
    print(f"4. x_mse = ||v|| Pi^T y_hat = {np.round(x_mse, 4)}"
          f"  rel. error = {np.linalg.norm(V - x_mse) / norm:.4f}  <q, x_mse> = {Q @ x_mse:.4f}")
    r = V - x_mse
    print(f"5. r = v - x_mse = {np.round(r, 4)}  ||r|| = {np.linalg.norm(r):.4f}  <q, r> = {Q @ r:.4f}")
    sr, sq = S_TQ @ r, S_TQ @ Q
    corr = qjl_correction(S_TQ, Q, r)
    print(f"6. S.r = {np.round(sr, 3)}  signs = {np.sign(sr)}  S.q = {sq}"
          f"  <S.q, sign> = {sq @ np.sign(sr):.2f}")
    print(f"7. correction = {corr:.4f}  estimate = {Q @ x_mse + corr:.4f}  (exact {Q @ V:.0f})")

    rng = np.random.default_rng(0)
    for m in (4, 64, 1024):
        est = [Q @ x_mse + qjl_correction(rng.standard_normal((m, D)), Q, r) for _ in range(2000)]
        print(f"   same rotation, 2,000 draws of S with m = {m}: mean {np.mean(est):.2f}, std {np.std(est):.2f}")
    mse_only, prod = [], []
    u = V / norm
    for _ in range(20_000):
        p = haar_rotation(D, rng)
        u_mse = p.T @ c[np.searchsorted(bounds, p @ u)]
        mse_only.append(norm * (Q @ u_mse))
        prod.append(norm * (Q @ u_mse + qjl_correction(rng.standard_normal((D, D)), Q, u - u_mse)))
    print(f"   20,000 random rotations: MSE stage alone mean {np.mean(mse_only):.2f} (std {np.std(mse_only):.2f}),"
          f" with QJL mean {np.mean(prod):.2f} (std {np.std(prod):.2f})")


if __name__ == "__main__":
    qjl()
    polarquant()
    turboquant()
