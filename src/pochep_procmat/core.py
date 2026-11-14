"""Matrix decompositions and solvers for the linear model M @ t = p.

Every public function validates its inputs and raises ``TypeError`` or
``ValueError`` on a wrong type, shape or non-finite value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_matrix(M: ArrayLike, name: str = "M") -> FloatArray:
    """Return ``M`` as a finite two-dimensional float64 array.

    Raises ``TypeError`` if ``M`` cannot be converted to floats and ``ValueError``
    if it is not two-dimensional, is empty or contains NaN or infinite values.
    """
    try:
        arr = np.asarray(M, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float array") from exc
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {arr.ndim} dimensions")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def as_vector(p: ArrayLike, name: str = "p", length: int | None = None) -> FloatArray:
    """Return ``p`` as a finite one-dimensional float64 array of the given length."""
    try:
        arr = np.asarray(p, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to a float array") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {arr.ndim} dimensions")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _check_pair(M: ArrayLike, p: ArrayLike) -> tuple[FloatArray, FloatArray]:
    m = as_matrix(M)
    vec = as_vector(p, length=m.shape[0])
    return m, vec


def qr_decompose(M: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Reduced QR decomposition ``M = Q R`` with ``Q`` of shape (T, r) and ``R`` of shape (r, r)."""
    m = as_matrix(M)
    q, r = np.linalg.qr(m, mode="reduced")
    return q, r


def svd_decompose(M: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Thin singular value decomposition ``M = U diag(s) V^T``.

    Returns ``(U, s, Vt)`` with ``s`` sorted in decreasing order.
    """
    m = as_matrix(M)
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    return u, s, vt


def effective_rank(M: ArrayLike, rtol: float = 1e-10) -> int:
    """Number of singular values larger than ``rtol`` times the largest one."""
    m = as_matrix(M)
    s = np.linalg.svd(m, compute_uv=False)
    if s[0] == 0.0:
        return 0
    return int(np.sum(s > rtol * s[0]))


def solve_qr(M: ArrayLike, p: ArrayLike) -> FloatArray:
    """Least-squares solution of ``M t = p`` through the QR decomposition.

    Requires ``M`` to have full column rank. Raises ``ValueError`` otherwise.
    """
    m, vec = _check_pair(M, p)
    if effective_rank(m) < m.shape[1]:
        raise ValueError("QR solver needs a full column rank matrix, use the svd method instead")
    q, r = np.linalg.qr(m, mode="reduced")
    return np.linalg.solve(r, q.T @ vec)


def solve_svd(M: ArrayLike, p: ArrayLike, rcond: float = 1e-10) -> FloatArray:
    """Minimum-norm least-squares solution through the truncated pseudo-inverse.

    Singular values below ``rcond`` times the largest one are treated as zero.
    """
    m, vec = _check_pair(M, p)
    if rcond < 0:
        raise ValueError("rcond must be non-negative")
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    keep = s > rcond * s[0] if s[0] > 0 else np.zeros_like(s, dtype=bool)
    coef = np.zeros(s.shape[0])
    coef[keep] = (u[:, keep].T @ vec) / s[keep]
    return vt.T @ coef


def solve_normal_cholesky(M: ArrayLike, p: ArrayLike, ridge: float = 0.0) -> FloatArray:
    """Solve ``(M^T M + ridge I) t = M^T p`` with a Cholesky factorisation.

    A positive ``ridge`` adds Tikhonov regularisation and makes the system solvable
    for rank-deficient ``M``. With ``ridge = 0`` the matrix must be positive definite,
    otherwise ``ValueError`` is raised.
    """
    m, vec = _check_pair(M, p)
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    gram = m.T @ m + ridge * np.eye(m.shape[1])
    try:
        chol = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as exc:
        raise ValueError("normal matrix is not positive definite, use ridge > 0") from exc
    y = np.linalg.solve(chol, m.T @ vec)
    return np.linalg.solve(chol.T, y)


def solve_nnls(
    M: ArrayLike,
    p: ArrayLike,
    *,
    max_iter: int = 20000,
    tol: float = 1e-12,
) -> FloatArray:
    """Non-negative least squares ``min ||M t - p||`` subject to ``t >= 0``.

    Solved by projected gradient descent with the step ``1 / L``, where ``L`` is the
    largest eigenvalue of ``M^T M``. Stops when the relative change of ``t`` is below ``tol``.
    """
    m, vec = _check_pair(M, p)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    gram = m.T @ m
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    rhs = m.T @ vec
    t = np.zeros(m.shape[1])
    if lipschitz == 0.0:
        return t
    for _ in range(max_iter):
        grad = gram @ t - rhs
        t_new = np.maximum(0.0, t - grad / lipschitz)
        change = np.linalg.norm(t_new - t)
        t = t_new
        if change <= tol * max(1.0, np.linalg.norm(t)):
            break
    return t


def nmf(
    M: ArrayLike,
    rank: int,
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
    seed: int = 0,
    eps: float = 1e-12,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Non-negative matrix factorisation ``M ~ W H`` by Lee and Seung multiplicative updates.

    ``M`` must be non-negative. Returns ``(W, H, errors)`` where ``W`` has shape (T, rank),
    ``H`` has shape (rank, r) and ``errors`` is the Frobenius error after each iteration.
    The random initialisation is controlled by ``seed``.
    """
    m = as_matrix(M)
    if np.any(m < 0):
        raise ValueError("NMF needs a non-negative matrix")
    if not isinstance(rank, (int, np.integer)) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if rank > min(m.shape):
        raise ValueError("rank must not exceed min(M.shape)")
    rng = np.random.default_rng(seed)
    scale = np.sqrt(m.mean() / rank) if m.mean() > 0 else 1.0
    w = np.abs(rng.standard_normal((m.shape[0], rank))) * scale + eps
    h = np.abs(rng.standard_normal((rank, m.shape[1]))) * scale + eps
    errors: list[float] = []
    previous = np.inf
    for _ in range(max_iter):
        h *= (w.T @ m) / (w.T @ w @ h + eps)
        w *= (m @ h.T) / (w @ h @ h.T + eps)
        err = float(np.linalg.norm(m - w @ h))
        errors.append(err)
        if abs(previous - err) <= tol * max(1.0, err):
            break
        previous = err
    return w, h, np.asarray(errors)


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine of the angle between two vectors of the same length, in [-1, 1]."""
    va = as_vector(a, name="a")
    vb = as_vector(b, name="b", length=va.shape[0])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(va @ vb / (na * nb))
