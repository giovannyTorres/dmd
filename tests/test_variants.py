"""Numerical recovery tests for the non-autonomous and preprocessing variants."""

import numpy as np
import pytest

from modaldyn import DMDc, HankelDMD, RobustDMD, delay_embedding


def controlled_sequence(seed=4):
    rng = np.random.default_rng(seed)
    a = np.array([[0.9, 0.08], [-0.03, 0.8]])
    b = np.array([[0.3], [-0.4]])
    controls = rng.normal(size=(1, 100))
    x = np.zeros((2, 101))
    x[:, 0] = [1, -0.5]
    for k in range(100):
        x[:, k + 1] = a @ x[:, k] + b @ controls[:, k]
    return a, b, x, controls


def test_dmdc_recovers_known_state_and_control_operators():
    a, b, x, u = controlled_sequence()
    model = DMDc().fit(x, u)
    np.testing.assert_allclose(model.A_, a, atol=2e-14)
    np.testing.assert_allclose(model.B_, b, atol=2e-14)
    np.testing.assert_allclose(model.reconstruct(), x, atol=2e-13)
    future_u = np.array([[0.2, 0.8, -0.4]])
    expected = x[:, -1]
    future = []
    for control in future_u.T:
        expected = a @ expected + b @ control
        future.append(expected)
    np.testing.assert_allclose(model.forecast(future_u), np.array(future).T, atol=2e-14)
    assert model.diagnostics()["relative_residual"] < 1e-13


def test_dmdc_known_control_and_explicit_pairs():
    a, b, x, u = controlled_sequence()
    model = DMDc(rank=2).fit(x[:, :-1], u, x[:, 1:], B=b)
    np.testing.assert_allclose(model.A_, a, atol=2e-14)
    np.testing.assert_allclose(model.predict(x[:, 10], u[:, 10]), x[:, 11], atol=2e-14)
    with pytest.raises(ValueError, match="unordered"):
        model.reconstruct()


def test_dmdc_fit_owns_snapshots_controls_and_known_B():
    _, b, x, u = controlled_sequence()
    model = DMDc(rank=1.0).fit(x, u, B=b)
    reconstructed = model.reconstruct().copy()
    forecast = model.forecast(np.zeros((1, 3))).copy()
    x[:] = 99
    u[:] = 99
    b[:] = 99
    np.testing.assert_allclose(model.reconstruct(), reconstructed)
    np.testing.assert_allclose(model.forecast(np.zeros((1, 3))), forecast)


def test_failed_refits_invalidate_previous_variant_fit():
    _, _, x, u = controlled_sequence()
    controlled = DMDc().fit(x, u)
    with pytest.raises(ValueError):
        controlled.fit(x, u[:, :-1])
    with pytest.raises(RuntimeError):
        controlled.reconstruct()
    hankel = HankelDMD(delays=3, rank=2).fit(x)
    with pytest.raises(ValueError):
        hankel.fit(x[:, :2])
    with pytest.raises(RuntimeError):
        hankel.reconstruct()
    robust = RobustDMD(rank=2).fit(x)
    with pytest.raises(ValueError):
        robust.fit(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        robust.reconstruct()


def test_dmdc_validates_control_alignment():
    _, _, x, u = controlled_sequence()
    with pytest.raises(ValueError, match="one column per transition"):
        DMDc().fit(x, u[:, :-1])
    with pytest.raises(RuntimeError, match="fit"):
        DMDc().forecast(u)
    with pytest.raises(ValueError, match="rank"):
        DMDc(rank=-1).fit(x, u)


def test_dmdc_low_residual_does_not_hide_unidentifiable_collinear_controls():
    x = (2.0 ** np.arange(12))[None, :]
    controls = x[:, :-1].copy()
    model = DMDc().fit(x, controls)
    diagnostics = model.diagnostics()
    assert diagnostics["relative_residual"] < 1e-13
    assert diagnostics["full_regression_rank"] == 1
    assert diagnostics["required_regression_rank"] == 2
    assert not diagnostics["identifiable"]
    assert any("Insufficient" in note for note in diagnostics["notes"])
    assert any("small training residual" in note for note in diagnostics["notes"])


def test_dmdc_diagnostics_distinguish_known_B_and_rank_truncation():
    _, b, x, controls = controlled_sequence()
    known = DMDc().fit(x, controls, B=b).diagnostics()
    assert known["full_regression_rank"] == known["required_regression_rank"] == 2
    assert known["identifiable"]
    complete = DMDc().fit(x, controls).diagnostics()
    assert complete["full_regression_rank"] == complete["required_regression_rank"] == 3
    assert complete["identifiable"]
    truncated = DMDc(rank=1).fit(x, controls).diagnostics()
    assert truncated["full_regression_rank"] == 3
    assert not truncated["identifiable"]
    assert any("rank truncation" in note for note in truncated["notes"])


def test_hankel_recovers_two_frequencies_from_one_sensor():
    t = np.arange(140)
    signal = (np.sin(0.31 * t) + 0.4 * np.cos(0.73 * t))[None, :]
    model = HankelDMD(delays=20, rank=4).fit(signal[:, :100])
    np.testing.assert_allclose(model.reconstruct(), signal[:, :100], atol=3e-12)
    np.testing.assert_allclose(model.forecast(40), signal[:, 100:], atol=3e-12)
    np.testing.assert_allclose(model.predict(signal[:, 80:100]), signal[:, 100], atol=3e-12)
    assert model.modes_.shape == (1, 4)
    assert model.embedded_modes_.shape == (20, 4)


def test_delay_embedding_and_validation():
    x = np.arange(12).reshape(2, 6)
    embedded = delay_embedding(x, 3)
    assert embedded.shape == (6, 4)
    np.testing.assert_array_equal(embedded[:, 0], [0, 6, 1, 7, 2, 8])
    with pytest.raises(ValueError, match="delays"):
        HankelDMD(delays=6).fit(x)
    with pytest.raises(RuntimeError, match="fit"):
        HankelDMD().reconstruct()


def test_rpca_decomposes_sparse_corruption_and_recovers_dynamics():
    rng = np.random.default_rng(12)
    q, _ = np.linalg.qr(rng.normal(size=(35, 2)))
    t = np.arange(90)
    clean = q @ np.vstack([np.cos(0.2 * t), np.sin(0.2 * t)])
    sparse = np.zeros_like(clean)
    support = rng.random(clean.shape) < 0.015
    sparse[support] = rng.normal(0, 3, support.sum())
    model = RobustDMD(rank=2).fit(clean + sparse)
    assert model.rpca_converged_
    assert model.rpca_residual_ < 1e-7
    assert np.linalg.norm(model.low_rank_ - clean) / np.linalg.norm(clean) < 2e-4
    np.testing.assert_allclose(model.low_rank_ + model.sparse_, clean + sparse, atol=1e-5)
    assert np.max(np.abs(np.sort_complex(model.eigenvalues_) - np.sort_complex(np.exp(1j * np.array([-0.2, 0.2]))))) < 2e-4


def test_rpca_warns_when_iteration_budget_is_exhausted():
    rng = np.random.default_rng(8)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model = RobustDMD(rank=2, rpca_max_iter=1).fit(rng.normal(size=(6, 20)))
    assert not model.rpca_converged_


@pytest.mark.parametrize("kind", ["dmdc", "hankel", "robust"])
def test_variant_torch_cpu_matches_numpy(kind):
    torch = pytest.importorskip("torch")
    a, b, x, u = controlled_sequence()
    if kind == "dmdc":
        model = DMDc(backend="torch", device="cpu").fit(torch.tensor(x), torch.tensor(u))
        np.testing.assert_allclose(model.A_.cpu().numpy(), a, atol=2e-14)
        np.testing.assert_allclose(model.B_.cpu().numpy(), b, atol=2e-14)
        np.testing.assert_allclose(model.reconstruct().cpu().numpy(), x, atol=2e-13)
    elif kind == "hankel":
        signal = np.sin(0.31 * np.arange(80))[None, :]
        model = HankelDMD(delays=10, rank=2, backend="torch", device="cpu").fit(torch.tensor(signal))
        np.testing.assert_allclose(model.reconstruct().cpu().numpy(), signal, atol=3e-12)
    else:
        rng = np.random.default_rng(12)
        q, _ = np.linalg.qr(rng.normal(size=(20, 2)))
        clean = q @ np.vstack([np.cos(0.2 * np.arange(50)), np.sin(0.2 * np.arange(50))])
        model = RobustDMD(rank=2, backend="torch", device="cpu").fit(torch.tensor(clean))
        assert model.rpca_converged_
        assert torch.isfinite(model.low_rank_).all()


def test_complex_torch_rpca_and_control_recovery():
    torch = pytest.importorskip("torch")
    t = torch.arange(60, dtype=torch.float64)
    spatial = torch.tensor([[1 + 1j], [0.3 - 0.8j], [-0.6 + 0.2j]], dtype=torch.complex128)
    clean = spatial @ torch.exp((0.2j * t)[None, :])
    model = RobustDMD(rank=1, backend="torch").fit(clean)
    assert model.rpca_converged_
    assert torch.isfinite(model.low_rank_).all()
    np.testing.assert_allclose(model.eigenvalues_.numpy(), [np.exp(0.2j)], atol=2e-7)
    a, b, x, u = controlled_sequence()
    controlled = DMDc(backend="torch").fit(torch.tensor(x, dtype=torch.complex128), torch.tensor(u))
    np.testing.assert_allclose(controlled.A_.numpy(), a, atol=2e-14)
    np.testing.assert_allclose(controlled.B_.numpy(), b, atol=2e-14)


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["dmdc", "hankel", "robust"])
def test_variant_cuda_recovery_and_device_residency(kind):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    options = {"backend": "torch", "device": "cuda"}
    if kind == "dmdc":
        a, b, x, u = controlled_sequence()
        model = DMDc(**options).fit(x, u)
        assert model.A_.is_cuda and model.B_.is_cuda
        assert model.forecast(np.zeros((1, 4))).is_cuda
        np.testing.assert_allclose(model.A_.cpu().numpy(), a, atol=2e-12)
        np.testing.assert_allclose(model.B_.cpu().numpy(), b, atol=2e-12)
        np.testing.assert_allclose(model.reconstruct().cpu().numpy(), x, atol=2e-11)
    elif kind == "hankel":
        signal = np.sin(0.31 * np.arange(100))[None, :]
        model = HankelDMD(delays=10, rank=2, **options).fit(signal[:, :80])
        assert model.embedded_modes_.is_cuda
        assert model.forecast(20).is_cuda
        np.testing.assert_allclose(model.reconstruct().cpu().numpy(), signal[:, :80], atol=3e-10)
        np.testing.assert_allclose(model.forecast(20).cpu().numpy(), signal[:, 80:], atol=3e-10)
    else:
        rng = np.random.default_rng(12)
        q, _ = np.linalg.qr(rng.normal(size=(35, 2)))
        t = np.arange(90)
        clean = q @ np.vstack([np.cos(0.2 * t), np.sin(0.2 * t)])
        corruption = np.zeros_like(clean)
        support = rng.random(clean.shape) < 0.015
        corruption[support] = rng.normal(0, 3, support.sum())
        model = RobustDMD(rank=2, **options).fit(clean + corruption)
        assert model.low_rank_.is_cuda and model.sparse_.is_cuda
        assert model.forecast(4).is_cuda
        assert model.rpca_converged_
        recovered = model.low_rank_.cpu().numpy()
        assert np.linalg.norm(recovered - clean) / np.linalg.norm(clean) < 2e-4
