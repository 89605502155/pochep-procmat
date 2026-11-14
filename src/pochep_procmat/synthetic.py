"""Synthetic process memory profiles for tests and the demo notebook.

The generator produces ``r`` non-negative profiles over ``T`` timestamps. Each profile
is a block of memory that an application holds while it is running and releases after
it stops, plus a small background noise. Nothing here models a real operating system;
the output is synthetic by construction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_process_profiles(
    n_times: int,
    n_processes: int,
    *,
    seed: int = 0,
    noise: float = 0.02,
    mean_block_mb: float = 150.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(M, t_true, p)`` with ``p = M @ t_true + noise``.

    ``M`` has shape (n_times, n_processes): column ``j`` is the memory profile of process ``j``
    in megabytes. ``t_true`` holds the true per-process weights, close to 1, and ``p`` is the
    observed total memory with additive Gaussian noise of relative size ``noise``.
    """
    if n_times < 2 or n_processes < 1:
        raise ValueError("need at least 2 timestamps and 1 process")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if mean_block_mb <= 0:
        raise ValueError("mean_block_mb must be positive")
    rng = np.random.default_rng(seed)
    profiles = np.zeros((n_times, n_processes))
    for j in range(n_processes):
        n_blocks = rng.integers(1, 4)
        for _ in range(n_blocks):
            start = rng.integers(0, n_times - 1)
            length = rng.integers(1, max(2, n_times // 3))
            stop = min(n_times, start + length)
            height = rng.uniform(0.2, 1.8) * mean_block_mb
            profiles[start:stop, j] += height
    t_true = rng.uniform(0.9, 1.1, size=n_processes)
    clean = profiles @ t_true
    scale = np.linalg.norm(clean) / np.sqrt(n_times) if n_times else 1.0
    p = clean + noise * scale * rng.standard_normal(n_times)
    return profiles, t_true, p
