import json

import numpy as np
import pytest

from modaldyn.streaming import StreamingDMD


def test_recovers_linear_operator_from_independent_batches():
    rng = np.random.default_rng(7)
    a = np.array([[0.8, -0.2], [0.1, 0.95]])
    x = rng.normal(size=(2, 60))
    y = a @ x
    model = StreamingDMD()
    model.partial_fit(x[:, :17], y[:, :17])
    model.partial_fit(x[:, 17:], y[:, 17:]).finalize()
    np.testing.assert_allclose(model.operator_, a, atol=1e-12)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-12)
    expected = np.column_stack([np.linalg.matrix_power(a, i) @ x[:, 0] for i in range(1, 6)])
    np.testing.assert_allclose(model.forecast(x[:, 0], 5), expected, atol=1e-12)
    assert model.forecast(x[:, :1], 0).shape == (2, 0)
    assert model.n_pairs_seen_ == 60
    json.dumps(model.diagnostics(), allow_nan=False)


def test_unpaired_batches_do_not_infer_boundary_transition():
    first = np.array([[1.0, 2.0, 4.0]])
    second = np.array([[100.0, 300.0]])
    model = StreamingDMD().partial_fit(first).partial_fit(second).finalize()
    x = np.array([[1.0, 2.0, 100.0]])
    y = np.array([[2.0, 4.0, 300.0]])
    np.testing.assert_allclose(model.operator_, y @ np.linalg.pinv(x))
    assert model.n_pairs_seen_ == 3


def test_forgetting_factor_discounts_per_batch():
    model = StreamingDMD(forgetting_factor=0.5)
    model.partial_fit(np.array([[1.0, 2.0]]), np.array([[2.0, 4.0]]))
    model.partial_fit(np.array([[3.0]]), np.array([[12.0]])).finalize()
    np.testing.assert_allclose(model.operator_, [[(0.5 * 10 + 36) / (0.5 * 5 + 9)]])
    assert model.effective_pairs_ == 2.0


def test_complex_snapshots_and_rank_truncation():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3, 15)) + 1j * rng.normal(size=(3, 15))
    a = np.diag([0.8 + 0.2j, 0.9 - 0.1j, 0.5])
    model = StreamingDMD().fit(x, a @ x)
    np.testing.assert_allclose(model.operator_, a, atol=1e-12)
    low = StreamingDMD(rank=2).fit(x, a @ x)
    assert low.rank_ == 2
    assert low.modes_.shape == (3, 3)
    assert np.linalg.matrix_rank(low.operator_) == 2


def test_input_mutations_and_invalid_updates_preserve_accumulated_state():
    x = np.array([[1.0, 2.0, 4.0]])
    model = StreamingDMD().partial_fit(x)
    x[:] = 99
    model.finalize()
    np.testing.assert_allclose(model.operator_, [[2.0]])
    with pytest.raises(ValueError, match="finite"):
        model.partial_fit(np.array([[1.0, np.nan]]))
    np.testing.assert_allclose(model.predict(np.array([2.0])), [4.0])
    with pytest.raises(ValueError, match="feature count"):
        model.partial_fit(np.ones((2, 3)))
    assert model.n_batches_ == 1
    with pytest.raises(ValueError):
        model.fit(np.zeros((1, 3)))
    np.testing.assert_allclose(model.operator_, [[2.0]])


def test_fit_resets_counts_and_updates_require_finalization():
    model = StreamingDMD().fit(np.array([[1.0, 2.0, 4.0]]))
    model.partial_fit(np.array([[3.0, 6.0]]))
    with pytest.raises(RuntimeError, match="finalize"):
        model.predict(np.array([1.0]))
    model.finalize()
    assert model.n_batches_ == 2
    model.fit(np.array([[1.0, 3.0]]))
    assert model.n_batches_ == 1
    assert model.n_pairs_seen_ == 1
    np.testing.assert_allclose(model.operator_, [[3.0]])


@pytest.mark.parametrize("kwargs", [
    {"rank": 0}, {"rank": True}, {"rank": 0.5},
    {"dt": 0}, {"dt": np.nan},
    {"forgetting_factor": 0}, {"forgetting_factor": 1.1},
    {"rcond": -1}, {"backend": "unknown"},
])
def test_invalid_constructor_parameters(kwargs):
    with pytest.raises(ValueError):
        StreamingDMD(**kwargs)


def test_invalid_states_and_pairs():
    model = StreamingDMD()
    with pytest.raises(RuntimeError):
        model.finalize()
    with pytest.raises(RuntimeError):
        model.predict(np.ones(2))
    for x, y in [(np.ones(3), None), (np.ones((1, 1)), None),
                 (np.ones((1, 2)), np.ones((2, 2))), (np.ones((0, 3)), None)]:
        with pytest.raises(ValueError):
            model.partial_fit(x, y)
    with pytest.raises(ValueError, match="rank"):
        StreamingDMD(rank=2).fit(np.ones((1, 3)))
    with pytest.raises(ValueError, match="zero numerical rank"):
        model.fit(np.zeros((2, 3)))
    model.fit(np.array([[1.0, 2.0, 4.0]]))
    for state in [np.ones(2), np.ones((1, 0)), np.array([np.inf])]:
        with pytest.raises(ValueError):
            model.predict(state)
    for horizon in [-1, 0.1, True]:
        with pytest.raises(ValueError):
            model.forecast(np.ones(1), horizon)
    with pytest.raises(ValueError):
        model.forecast(np.ones((1, 2)), 2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_backend_parity_and_mixed_input_types(device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(12)
    x = rng.normal(size=(2, 20))
    a = np.array([[0.9, -0.1], [0.1, 0.9]])
    model = StreamingDMD(backend="torch", device=device)
    model.partial_fit(torch.tensor(x[:, :10], device=device), a @ x[:, :10])
    model.partial_fit(x[:, 10:], a @ x[:, 10:]).finalize()
    np.testing.assert_allclose(model.operator_.detach().cpu().numpy(), a, atol=1e-12)
    assert model.predict(x).device.type == device
    complex_state = x[:, 0] + 1j
    result = model.predict(complex_state).detach().cpu().numpy()
    np.testing.assert_allclose(result, a @ complex_state, atol=1e-12)
