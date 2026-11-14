# pochep-procmat

Matrix decompositions for the process memory model `M t = p`.

Rows of `M` are timestamps and columns are processes. `p` is the total memory measured at each
timestamp and `t` is the vector of per-process weights. The package solves this linear system
with QR, SVD, Cholesky normal equations, non-negative least squares (NNLS), and fits NMF
factors `M ~ W H`. It also provides a synthetic generator of process profiles for tests.

Russian version: [README.ru.md](README.ru.md). Documentation: [docs/en](docs/en/theory.md), [docs/ru](docs/ru/theory.md).
Demo notebook with strengths and weaknesses: https://github.com/89605502155/pochep-procmat-demo

## Install

```bash
pip install pochep-procmat
```

Requires Python 3.11 to 3.14 and NumPy 2.0 or newer. Released on PyPI: https://pypi.org/project/pochep-procmat/

## Example

```python
from pochep_procmat import ProcessMemoryModel, make_process_profiles

M, t_true, p = make_process_profiles(n_times=120, n_processes=4, seed=1, noise=0.01)
model = ProcessMemoryModel("svd")
result = model.fit(M, p)
print(result.weights, result.relative_residual, result.rank)
```

## Methods

| Method | Function | Needs | Output |
|---|---|---|---|
| `qr` | `solve_qr` | full column rank | least-squares weights |
| `svd` | `solve_svd` | none, truncation by `rcond` | minimum-norm weights |
| `cholesky` | `solve_normal_cholesky` | positive definite `M^T M`, or `ridge > 0` | weights |
| `nnls` | `solve_nnls` | none | non-negative weights |
| `nmf` | `nmf`, then NNLS | non-negative `M` and `p` | non-negative weights from `W H` |

## Licence

MIT. Authors: Andrey O. Ferubko, Oleg D. Kazakov. How to cite: see `CITATION.cff`.
