# pochep_procmat API (version 0.0.1)

- `ProcessMemoryModel(method="svd", *, rcond=1e-10, ridge=0.0, nmf_rank=None, seed=0)`: the model `M t = p`.
  `fit(M, p)` returns a `FitResult`, `predict(M)` returns `M @ t`, property `weights`.
- `FitResult`: `method`, `weights`, `residual_norm`, `relative_residual`, `rank`, `seconds`.
- `solve_qr(M, p)`, `solve_svd(M, p, rcond=1e-10)`, `solve_normal_cholesky(M, p, ridge=0.0)`,
  `solve_nnls(M, p, *, max_iter=20000, tol=1e-12)`.
- `qr_decompose(M)`, `svd_decompose(M)`, `nmf(M, rank, *, max_iter=500, tol=1e-6, seed=0)`.
- `effective_rank(M, rtol=1e-10)`, `cosine_similarity(a, b)`.
- `make_process_profiles(n_times, n_processes, *, seed=0, noise=0.02, mean_block_mb=150.0)`
  returns `(M, t_true, p)` of synthetic data.
- `as_matrix`, `as_vector`: input checks. A wrong type raises `TypeError`, a wrong shape, an empty array
  or NaN raises `ValueError`.
