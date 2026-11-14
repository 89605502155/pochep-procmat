"""Model M t = p: per-process weights recovered from a matrix of process memory profiles.

Rows of ``M`` are timestamps, columns are processes. ``p`` is the measured total
memory at each timestamp and ``t`` is the vector of per-process weights.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from pochep_procmat.core import (
    FloatArray,
    _check_pair,
    as_matrix,
    effective_rank,
    nmf,
    solve_nnls,
    solve_normal_cholesky,
    solve_qr,
    solve_svd,
)

METHODS = ("qr", "svd", "cholesky", "nnls", "nmf")


@dataclass(frozen=True)
class FitResult:
    """Outcome of one fit: weights, residual and cost of the chosen method."""

    method: str
    weights: FloatArray
    residual_norm: float
    relative_residual: float
    rank: int
    seconds: float


class ProcessMemoryModel:
    """Linear model ``M t = p`` solved by one of several decompositions.

    Parameters
    ----------
    method:
        One of ``"qr"``, ``"svd"``, ``"cholesky"``, ``"nnls"`` or ``"nmf"``.
    rcond:
        Relative cut-off for singular values in the ``"svd"`` method.
    ridge:
        Tikhonov term used by the ``"cholesky"`` method.
    nmf_rank:
        Number of latent components for the ``"nmf"`` method. The weights are then
        obtained from the non-negative coefficients of the rank-``nmf_rank`` model.
    seed:
        Seed for the random initialisation of NMF.
    """

    def __init__(
        self,
        method: str = "svd",
        *,
        rcond: float = 1e-10,
        ridge: float = 0.0,
        nmf_rank: int | None = None,
        seed: int = 0,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        if rcond < 0:
            raise ValueError("rcond must be non-negative")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        self.method = method
        self.rcond = rcond
        self.ridge = ridge
        self.nmf_rank = nmf_rank
        self.seed = seed
        self._weights: FloatArray | None = None

    @property
    def weights(self) -> FloatArray:
        """Weights found by the last ``fit``. Raises ``RuntimeError`` before a fit."""
        if self._weights is None:
            raise RuntimeError("the model is not fitted yet")
        return self._weights

    def fit(self, M: ArrayLike, p: ArrayLike) -> FitResult:
        """Estimate ``t`` from ``M`` and ``p`` and return a :class:`FitResult`."""
        m, vec = _check_pair(M, p)
        start = time.perf_counter()
        if self.method == "qr":
            t = solve_qr(m, vec)
        elif self.method == "svd":
            t = solve_svd(m, vec, rcond=self.rcond)
        elif self.method == "cholesky":
            t = solve_normal_cholesky(m, vec, ridge=self.ridge)
        elif self.method == "nnls":
            t = solve_nnls(m, vec)
        else:
            t = self._fit_nmf(m, vec)
        seconds = time.perf_counter() - start
        self._weights = t
        residual = float(np.linalg.norm(m @ t - vec))
        norm_p = float(np.linalg.norm(vec))
        relative = residual / norm_p if norm_p > 0 else residual
        return FitResult(
            method=self.method,
            weights=t,
            residual_norm=residual,
            relative_residual=relative,
            rank=effective_rank(m),
            seconds=seconds,
        )

    def _fit_nmf(self, m: FloatArray, vec: FloatArray) -> FloatArray:
        if np.any(m < 0):
            raise ValueError("the nmf method needs a non-negative matrix")
        if np.any(vec < 0):
            raise ValueError("the nmf method needs a non-negative vector p")
        rank = self.nmf_rank if self.nmf_rank is not None else min(m.shape)
        w, h, _ = nmf(m, rank, seed=self.seed)
        # The weights are fitted against the rank-limited approximation W H, and the
        # non-negativity of t keeps them interpretable as process shares.
        return solve_nnls(w @ h, vec)

    def predict(self, M: ArrayLike) -> FloatArray:
        """Return ``M @ t`` for the weights found by the last fit."""
        t = self.weights
        m = as_matrix(M)
        if m.shape[1] != t.shape[0]:
            raise ValueError(f"M must have {t.shape[0]} columns, got {m.shape[1]}")
        return m @ t
