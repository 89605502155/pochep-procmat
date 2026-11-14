import numpy as np
import pytest

from pochep_procmat import METHODS, ProcessMemoryModel, make_process_profiles


def test_all_methods_fit_synthetic_data():
    m, t_true, p = make_process_profiles(120, 4, seed=5, noise=0.0)
    for method in METHODS:
        model = ProcessMemoryModel(method, nmf_rank=4)
        res = model.fit(m, p)
        assert res.method == method
        assert res.weights.shape == (4,)
        assert res.relative_residual < 0.2


def test_svd_and_qr_agree_on_full_rank():
    m, _, p = make_process_profiles(80, 3, seed=7, noise=0.01)
    a = ProcessMemoryModel("svd").fit(m, p).weights
    b = ProcessMemoryModel("qr").fit(m, p).weights
    assert np.allclose(a, b, atol=1e-8)


def test_predict_uses_fitted_weights():
    m, _, p = make_process_profiles(60, 3, seed=9, noise=0.0)
    model = ProcessMemoryModel("svd")
    model.fit(m, p)
    assert np.allclose(model.predict(m), m @ model.weights)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ProcessMemoryModel().predict(np.ones((3, 2)))


def test_bad_method_and_parameters():
    with pytest.raises(ValueError):
        ProcessMemoryModel("lu")
    with pytest.raises(ValueError):
        ProcessMemoryModel("svd", rcond=-1.0)
    with pytest.raises(ValueError):
        ProcessMemoryModel("cholesky", ridge=-1.0)


def test_fit_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ProcessMemoryModel("svd").fit(np.ones((5, 2)), np.ones(4))


def test_synthetic_generator_validates_input():
    with pytest.raises(ValueError):
        make_process_profiles(1, 3)
    with pytest.raises(ValueError):
        make_process_profiles(10, 3, noise=-0.1)


def test_synthetic_generator_is_reproducible():
    a = make_process_profiles(50, 3, seed=11)
    b = make_process_profiles(50, 3, seed=11)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
