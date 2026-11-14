# API pochep_procmat (версия 0.0.1)

- `ProcessMemoryModel(method="svd", *, rcond=1e-10, ridge=0.0, nmf_rank=None, seed=0)`: модель `M t = p`.
  Методы `fit(M, p)` возвращает `FitResult`, `predict(M)` возвращает `M @ t`, свойство `weights`.
- `FitResult`: `method`, `weights`, `residual_norm`, `relative_residual`, `rank`, `seconds`.
- `solve_qr(M, p)`, `solve_svd(M, p, rcond=1e-10)`, `solve_normal_cholesky(M, p, ridge=0.0)`,
  `solve_nnls(M, p, *, max_iter=20000, tol=1e-12)`.
- `qr_decompose(M)`, `svd_decompose(M)`, `nmf(M, rank, *, max_iter=500, tol=1e-6, seed=0)`.
- `effective_rank(M, rtol=1e-10)`, `cosine_similarity(a, b)`.
- `make_process_profiles(n_times, n_processes, *, seed=0, noise=0.02, mean_block_mb=150.0)`
  возвращает `(M, t_true, p)` синтетических данных.
- `as_matrix`, `as_vector`: проверка входа. Неверный тип вызывает `TypeError`, неверная форма,
  пустой массив или NaN вызывают `ValueError`.
