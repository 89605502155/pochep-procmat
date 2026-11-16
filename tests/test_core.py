import numpy as np
import pytest

from pochep_procmat import (
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


@pytest.fixture
def full_rank():
    rng = np.random.default_rng(1)
    m = rng.uniform(0.0, 1.0, size=(40, 5))
    t = rng.uniform(0.5, 1.5, size=5)
    return m, t, m @ t


def test_qr_reconstructs_matrix(full_rank):
    m, _, _ = full_rank
    q, r = qr_decompose(m)
    assert q.shape == (40, 5) and r.shape == (5, 5)
    assert np.allclose(q @ r, m)
    assert np.allclose(q.T @ q, np.eye(5))


def test_svd_reconstructs_and_sorts(full_rank):
    m, _, _ = full_rank
    u, s, vt = svd_decompose(m)
    assert np.allclose(u @ np.diag(s) @ vt, m)
    assert np.all(np.diff(s) <= 0)


def test_solvers_recover_exact_weights(full_rank):
    m, t, p = full_rank
    for solved in (solve_qr(m, p), solve_svd(m, p), solve_normal_cholesky(m, p), solve_nnls(m, p)):
        assert np.allclose(solved, t, atol=1e-6)


def test_svd_matches_lstsq_on_rank_deficient():
    m = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    p = np.array([1.0, 2.0, 0.5, 1.5])
    expected = np.linalg.lstsq(m, p, rcond=None)[0]
    assert np.allclose(solve_svd(m, p), expected, atol=1e-8)
    assert effective_rank(m) == 2


def test_qr_rejects_rank_deficient():
    m = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    with pytest.raises(ValueError, match="full column rank"):
        solve_qr(m, np.ones(3))


def test_cholesky_needs_ridge_when_singular():
    m = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="positive definite"):
        solve_normal_cholesky(m, np.ones(3))
    t = solve_normal_cholesky(m, np.ones(3), ridge=1e-6)
    assert np.allclose(m @ t, np.ones(3), atol=1e-3)


def test_nnls_respects_non_negativity():
    m = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    p = np.array([2.0, -1.0, 0.5])
    t = solve_nnls(m, p)
    assert np.all(t >= 0)
    assert t[1] == pytest.approx(0.0, abs=1e-8)


def test_nmf_is_non_negative_and_error_drops():
    rng = np.random.default_rng(2)
    m = rng.uniform(0.0, 1.0, size=(30, 6))
    w, h, errors = nmf(m, 3, max_iter=300, seed=4)
    assert np.all(w >= 0) and np.all(h >= 0)
    assert w.shape == (30, 3) and h.shape == (3, 6)
    assert errors[-1] <= errors[0]


def test_nmf_exact_for_low_rank_input():
    rng = np.random.default_rng(3)
    base = rng.uniform(0.1, 1.0, size=(20, 2)) @ rng.uniform(0.1, 1.0, size=(2, 5))
    w, h, errors = nmf(base, 2, max_iter=3000, tol=1e-10, seed=0)
    assert np.linalg.norm(base - w @ h) / np.linalg.norm(base) < 1e-2


@pytest.mark.parametrize(
    "bad",
    [np.zeros((0, 3)), np.ones(5), [[1.0, np.nan]], [["a", "b"]]],
)
def test_as_matrix_rejects_bad_input(bad):
    with pytest.raises((TypeError, ValueError)):
        as_matrix(bad)


def test_as_vector_checks_length():
    with pytest.raises(ValueError, match="length 3"):
        as_vector([1.0, 2.0], length=3)


def test_nmf_rejects_negative_and_bad_rank():
    with pytest.raises(ValueError):
        nmf(np.array([[1.0, -1.0]]), 1)
    with pytest.raises(ValueError):
        nmf(np.ones((3, 3)), 4)
    with pytest.raises(ValueError):
        nmf(np.ones((3, 3)), 0)


def test_cosine_similarity_bounds():
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])


def _kkt_ok(m, p, t, rtol=1e-8):
    w = m.T @ (p - m @ t)
    scale = np.abs(m).sum(axis=0).max() * max(1.0, np.abs(p).max())
    pos = t > 0
    return np.all(t >= 0) and np.all(np.abs(w[pos]) <= rtol * scale) and np.all(w[~pos] <= rtol * scale)


def _nnls_brute(m, p):
    import itertools

    n = m.shape[1]
    best, best_f = np.zeros(n), float(np.sum(p**2))
    for k in range(1, n + 1):
        for cols in itertools.combinations(range(n), k):
            sub = np.linalg.lstsq(m[:, cols], p, rcond=None)[0]
            if np.any(sub < 0):
                continue
            t = np.zeros(n)
            t[list(cols)] = sub
            f = float(np.sum((m @ t - p) ** 2))
            if f < best_f:
                best, best_f = t, f
    return best, best_f


@pytest.mark.parametrize("seed", range(20))
def test_nnls_matches_brute_force(seed):
    rng = np.random.default_rng(100 + seed)
    m = rng.normal(size=(15, 5))
    p = rng.normal(size=15)
    t = solve_nnls(m, p)
    _, f_best = _nnls_brute(m, p)
    assert float(np.sum((m @ t - p) ** 2)) == pytest.approx(f_best, rel=1e-9, abs=1e-9)
    assert _kkt_ok(m, p, t)


def test_nnls_converges_on_badly_scaled_columns():
    # Columns of very different magnitude, as RSS of processes in MB (regression for 0.0.1,
    # where projected gradient stopped far from the optimum).
    rng = np.random.default_rng(7)
    base = rng.uniform(0.5, 1.5, size=(400, 6)) + np.linspace(0, 1, 400)[:, None]
    m = base * np.array([10.0, 30.0, 300.0, 500.0, 4000.0, 9.0])
    t_true = np.array([0.6, 0.7, 1.3, 0.0, 1.0, 0.0])
    p = m @ t_true + rng.normal(scale=5.0, size=400)
    t = solve_nnls(m, p)
    assert _kkt_ok(m, p, t)
    t_ls = np.linalg.lstsq(m, p, rcond=None)[0]
    if np.all(t_ls >= 0):
        assert np.allclose(t, t_ls, rtol=1e-8, atol=1e-8)


def test_nnls_zero_rhs_and_validation():
    m = np.eye(3)
    assert np.all(solve_nnls(m, np.zeros(3)) == 0)
    assert np.allclose(solve_nnls(m, np.array([1.0, -2.0, 3.0])), [1.0, 0.0, 3.0])
    with pytest.raises(ValueError):
        solve_nnls(m, np.ones(3), max_iter=0)
    with pytest.raises(ValueError):
        solve_nnls(m, np.ones(3), tol=-1.0)
