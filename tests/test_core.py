"""Scientific checks against known operators, rather than implementation mirrors."""
import json

import numpy as np
import pytest

from modaldyn.core import DMD, FbDMD, RandomizedDMD, TLSqDMD


def trajectory(A, x0, length):
    out = [np.asarray(x0)]
    for _ in range(length - 1):
        out.append(A @ out[-1])
    return np.column_stack(out)


@pytest.fixture
def rotation():
    theta, radius = .23, .98
    A = radius * np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    X = trajectory(A, np.array([1., -.3]), 80)
    return A, X


@pytest.mark.parametrize("factory", [DMD, TLSqDMD, FbDMD, RandomizedDMD])
def test_variants_recover_diagonalizable_system(factory, rotation):
    A, X = rotation
    m = factory(rank=2, dt=.2).fit(X[:, :60])
    # Conjugate eigenvalues can have real parts differing by one ulp. Sorting
    # lexicographically makes their order depend on the LAPACK build.
    actual, expected = m.eigenvalues_, np.linalg.eigvals(A)
    np.testing.assert_allclose(actual[np.argsort(np.angle(actual))],
                               expected[np.argsort(np.angle(expected))], atol=2e-12)
    np.testing.assert_allclose(m.reconstruct(), X[:, :60], atol=2e-12)
    np.testing.assert_allclose(m.forecast(20), X[:, 60:], atol=3e-12)
    np.testing.assert_allclose(m.predict(X[:, :5]), A @ X[:, :5], atol=2e-12)
    np.testing.assert_allclose(np.sort(m.frequencies_), np.array([-.23, .23]) / (2 * np.pi * .2), atol=2e-12)
    np.testing.assert_allclose(m.growth_rates_, np.log(.98) / .2, atol=2e-12)
    assert m.diagnostics()["relative_one_step_error"] < 1e-11


def test_arbitrary_pairs_recover_nonnormal_complex_operator():
    rng = np.random.default_rng(771)
    A = np.array([[.9+.1j, .3-.2j], [0, -.6+.2j]])
    X = rng.normal(size=(2, 12)) + 1j * rng.normal(size=(2, 12))
    m = DMD(rank=2).fit(X, A @ X)
    unseen = rng.normal(size=(2, 7)) + 1j * rng.normal(size=(2, 7))
    np.testing.assert_allclose(m.predict(unseen), A @ unseen, atol=2e-12)
    np.testing.assert_allclose(m.predict(unseen[:, 0]), A @ unseen[:, 0], atol=2e-12)
    # Explicit pair fitting still rolls out from the first X column.
    np.testing.assert_allclose(m.reconstruct(4), trajectory(A, X[:, 0], 4), atol=2e-12)


def test_energy_rank_and_numerical_rank():
    X = np.diag([10., 1., 0.])
    Y = np.diag([.9, .8, .7]) @ X
    m = DMD(rank=.98).fit(X, Y)
    assert m.rank_ == 1
    assert m.numerical_rank_ == 2
    assert DMD(rank=1.).fit(X, Y).rank_ == 2
    assert DMD(rank=40).fit(X, Y).rank_ == 2
    assert m.diagnostics()["notes"]
    assert np.isfinite(m.reconstruct()).all()


def test_low_rank_embedding_and_seeded_randomized_svd():
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(40, 3)))
    latent = trajectory(np.diag([.99, .9, -.8]), np.ones(3), 50)
    X = Q @ latent
    before = np.random.get_state()
    a = RandomizedDMD(rank=3, oversampling=1, power_iters=2, seed=22).fit(X)
    after = np.random.get_state()
    b = RandomizedDMD(rank=3, oversampling=1, power_iters=2, seed=22).fit(X)
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(a.eigenvalues_, b.eigenvalues_)
    np.testing.assert_allclose(a.reconstruct(), X, atol=3e-11)


def test_rank_deficiency_rcond_and_zero_dynamics():
    X = np.diag([1., 1e-14])
    m = DMD(rcond=1e-10).fit(X, X)
    assert m.rank_ == 1
    z = DMD().fit(np.eye(2), np.zeros((2, 2)))
    np.testing.assert_allclose(z.predict(np.ones(2)), 0, atol=1e-15)
    np.testing.assert_allclose(z.reconstruct(1)[:, 0], [1., 0.], atol=1e-15)
    np.testing.assert_allclose(z.reconstruct(3)[:, 1:], 0, atol=1e-15)
    assert z.diagnostics()["relative_one_step_error"] is None
    json.dumps(z.diagnostics(), allow_nan=False)


def test_steps_shapes_and_training_mutation(rotation):
    _, original = rotation
    X = original.copy()
    m = DMD(rank=2).fit(X)
    reference = m.forecast(3).copy()
    X[:] = 100
    np.testing.assert_allclose(m.forecast(3), reference)
    assert m.forecast(0).shape == (2, 0)
    assert m.reconstruct(0).shape == (2, 0)
    np.testing.assert_allclose(m.reconstruct([3, 1, 3]), original[:, [3, 1, 3]], atol=1e-12)
    for steps in [-1, 1.2, [1.5], [-1, 3], [[1, 2]], True]:
        with pytest.raises(ValueError):
            m.reconstruct(steps)
    for steps in [-1, 1.1, True]:
        with pytest.raises(ValueError):
            m.forecast(steps)
    json.dumps(m.diagnostics(), allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"rank": 0}, {"rank": -1}, {"rank": True}, {"rank": 1.1},
    {"rank": "auto"}, {"rank": np.nan}, {"dt": 0}, {"dt": np.inf},
    {"tlsq_rank": -1}, {"power_iters": .5}, {"oversampling": False},
    {"rcond": 1}, {"rcond": np.nan}, {"randomized": True},
])
def test_invalid_constructor(kwargs):
    with pytest.raises(ValueError):
        DMD(**kwargs)


@pytest.mark.parametrize("X,Y", [
    (np.ones(3), None), (np.empty((0, 3)), None), (np.ones((2, 1)), None),
    (np.zeros((2, 3)), None), (np.array([[1., np.nan]]), None),
    (np.array([[1., np.inf]]), None), (np.ones((2, 3)), np.ones((3, 3))),
])
def test_invalid_fit(X, Y):
    with pytest.raises(ValueError):
        DMD().fit(X, Y)


def test_unfitted_failed_fit_and_predict_validation(rotation):
    model = DMD()
    for call in [lambda: model.predict(np.ones(2)), model.reconstruct,
                 lambda: model.forecast(1), model.diagnostics]:
        with pytest.raises(RuntimeError):
            call()
    model.fit(rotation[1])
    for X in [np.ones(3), np.ones((3, 5)), np.full(2, np.nan), 1.]:
        with pytest.raises(ValueError):
            model.predict(X)
    with pytest.raises(ValueError):
        model.fit(np.zeros((2, 3)))
    with pytest.raises(RuntimeError):
        model.forecast(1)


def test_forward_backward_singular_operator_rejected():
    with pytest.raises(ValueError, match="singular"):
        FbDMD().fit(np.eye(2), np.zeros((2, 2)))


def test_defective_operator_reports_modal_limitation():
    # A Jordan block cannot be represented by an invertible eigenvector basis.
    A = np.array([[.9, 1.], [0., .9]])
    info = DMD().fit(np.eye(2), A).diagnostics()
    assert info["mode_condition_number"] is None or info["mode_condition_number"] > 1e8
    assert any("ill-conditioned" in note for note in info["notes"])
    assert info["relative_one_step_error"] > .1
    json.dumps(info, allow_nan=False)


def test_randomized_rank_diagnostic_describes_sketch_scope():
    X = np.random.default_rng(10).normal(size=(5, 30))
    info = RandomizedDMD(rank=1, oversampling=0).fit(X).diagnostics()
    assert info["numerical_rank_scope"] == "randomized_sketch"
    assert any("omitted directions" in note for note in info["notes"])


def test_forward_backward_branch_ambiguity_is_reported():
    model = FbDMD().fit(np.array([[1., -1., 1., -1.]]))
    np.testing.assert_allclose(model.eigenvalues_, [1.])
    assert any("branch ambiguity" in note for note in model.diagnostics()["notes"])
