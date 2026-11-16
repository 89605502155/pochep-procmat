# Changelog

## 0.0.2

- Fixed `solve_nnls`: projected gradient descent stopped far from the optimum when columns of `M`
  differ in scale (real process RSS data). Replaced by the Lawson and Hanson active set method,
  which meets the KKT conditions and matches `scipy.optimize.nnls`.
- `max_iter` now limits outer iterations (default `3 * n`); `RuntimeError` if it is reached.
- New tests: brute force comparison, KKT check, badly scaled columns. 46 tests on Python 3.11–3.14.

## 0.0.1

- First release. QR, SVD, Cholesky normal equations, NNLS and NMF solvers for `M t = p`.
- `ProcessMemoryModel` with `fit`, `predict` and `FitResult`.
- Synthetic generator `make_process_profiles` for tests and the demo.
- Input validators raising `TypeError` or `ValueError`. 24 tests on Python 3.14.
