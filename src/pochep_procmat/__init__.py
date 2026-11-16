"""pochep_procmat: matrix decompositions for the process memory model M t = p."""

from pochep_procmat.core import (
    as_matrix,
    as_vector,
    cosine_similarity,
    effective_rank,
    nmf,
    qr_decompose,
    solve_nnls,
    solve_normal_cholesky,
    solve_qr,
    solve_svd,
    svd_decompose,
)
from pochep_procmat.model import METHODS, FitResult, ProcessMemoryModel
from pochep_procmat.synthetic import make_process_profiles

__version__ = "0.0.2"

__all__ = [
    "METHODS",
    "FitResult",
    "ProcessMemoryModel",
    "__version__",
    "as_matrix",
    "as_vector",
    "cosine_similarity",
    "effective_rank",
    "make_process_profiles",
    "nmf",
    "qr_decompose",
    "solve_nnls",
    "solve_normal_cholesky",
    "solve_qr",
    "solve_svd",
    "svd_decompose",
]
