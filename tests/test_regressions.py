"""Scientific regressions for precision and RPCA stopping behavior."""
import json

import numpy as np
import pytest

from modaldyn import DMD, RobustDMD


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_mixed_precision_pairs_preserve_resolvable_dynamics(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    x = np.diag(np.array([1.0, 1e-8], dtype=np.float32))
    operator = np.diag([0.9, 0.8])
    y = operator @ x.astype(np.float64)
    model = DMD(backend=backend).fit(x, y)
    assert model.rank_ == 2
    np.testing.assert_allclose(model.backend_.to_numpy(model.predict(np.eye(2))), operator, atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("complex_data", [False, True])
def test_robust_convergence_requires_component_stability(backend, complex_data):
    if backend == "torch":
        pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.normal(size=(24, 2)))
    t = np.arange(50)
    clean = q @ np.vstack([np.cos(0.2 * t), np.sin(0.2 * t)])
    sparse = np.zeros_like(clean)
    mask = rng.random(clean.shape) < 0.05
    sparse[mask] = rng.normal(0, 3, mask.sum())
    if complex_data:
        clean = clean * (1 + 0.5j)
        sparse = sparse * (1 - 0.25j)
    model = RobustDMD(rank=2, backend=backend).fit(clean + sparse)
    recovered = model.backend_.to_numpy(model.low_rank_)
    assert model.rpca_converged_
    assert model.rpca_error_ <= model.rpca_tol
    assert model.rpca_dual_error_ <= model.rpca_tol
    assert np.linalg.norm(recovered - clean) / np.linalg.norm(clean) < 1e-5
    json.dumps(model.diagnostics(), allow_nan=False)


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_diagnostics_are_invariant_under_snapshot_rescaling(scale):
    x = np.diag([1., 0.5]) * scale
    model = DMD().fit(x, np.diag([0.9, 0.8]) @ x)
    info = model.diagnostics()
    assert info["relative_one_step_error"] < 1e-12
    assert info["relative_initial_projection_error"] < 1e-12
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("kwargs", [{"dt": True}, {"dt": "1"}, {"rcond": True}, {"rcond": "1"}])
def test_invalid_numeric_options_fail_clearly(kwargs):
    with pytest.raises(ValueError):
        DMD(**kwargs)
