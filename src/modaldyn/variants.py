"""Delay embedding, controlled dynamics, and robust preprocessing.

Arrays use the convention ``(features, snapshots)`` throughout. All dense
linear algebra executes on the selected backend; diagnostics return scalars.
"""

from __future__ import annotations

import math
import warnings
from numbers import Integral, Real

import numpy as np

from ._backend import get_backend
from .core import DMD


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _matrix(value, backend, name):
    a = backend.asarray(value)
    if a.ndim != 2 or min(a.shape) < 1:
        raise ValueError(f"{name} must be a nonempty 2-D (features, snapshots) array")
    if not bool(backend.xp.isfinite(a).all()):
        raise ValueError(f"{name} must contain finite values")
    return a


def _promote(backend, *arrays):
    if backend.name == "numpy":
        dtype = np.result_type(*(x.dtype for x in arrays))
        return tuple(x.astype(dtype, copy=False) for x in arrays)
    dtype = arrays[0].dtype
    for x in arrays[1:]:
        dtype = backend.xp.promote_types(dtype, x.dtype)
    return tuple(x.to(dtype=dtype) for x in arrays)


def _rank_pinv(a, backend, rank):
    """Truncated pseudoinverse without constructing a full-state SVD basis."""
    u, s, vh = backend.svd(a)
    eps = (np.finfo(a.real.dtype).eps if backend.name == "numpy"
           else backend.xp.finfo(a.real.dtype).eps)
    cutoff = float(backend.scalar(s[0])) * max(a.shape) * eps
    numerical = int(backend.scalar((s > cutoff).sum()))
    if numerical == 0:
        raise ValueError("Regression input has numerical rank zero")
    if rank is None:
        r = numerical
    elif isinstance(rank, Integral) and not isinstance(rank, bool) and rank > 0:
        r = min(int(rank), numerical)
    elif isinstance(rank, Real) and not isinstance(rank, bool) and 0 < rank <= 1:
        energy = ((s[:numerical] / s[0]) ** 2).cumsum(0)
        r = min(int(backend.scalar((energy / energy[-1] < rank).sum())) + 1, numerical)
    else:
        raise ValueError("rank must be None, a positive integer, or energy fraction in (0, 1]")
    return (vh[:r].conj().T / s[:r]) @ u[:, :r].conj().T, r, numerical


def delay_embedding(X, delays=2, *, backend="auto", device=None):
    """Stack consecutive windows, oldest sample first, without circular wrap.

    An input of shape ``(n, m)`` produces ``(n * delays, m - delays + 1)``.
    This allocates the embedding; it is not an out-of-core algorithm.
    """
    delays = _positive_int(delays, "delays")
    b = get_backend(X, backend=backend, device=device)
    x = _matrix(X, b, "X")
    if x.shape[1] < delays:
        raise ValueError("X must have at least delays snapshots")
    width = x.shape[1] - delays + 1
    return b.concat([x[:, j:j + width] for j in range(delays)], axis=0)


class HankelDMD:
    """DMD on delay windows, with overlap-averaged physical reconstruction.

    ``modes_`` contains the first physical block of each embedded mode;
    ``embedded_modes_`` exposes the full delay-space modes. Forecasts start
    from the last observed window and return physical samples after training.
    Delay embedding increases memory by roughly ``delays``.
    """

    def __init__(self, delays=2, rank=None, **dmd_options):
        self.delays = _positive_int(delays, "delays")
        self.rank = rank
        self.dmd_options = dict(dmd_options)
        self._fitted = False

    def fit(self, X):
        self._fitted = False
        b = get_backend(X, backend=self.dmd_options.get("backend", "auto"),
                        device=self.dmd_options.get("device"))
        x = _matrix(X, b, "X")
        if x.shape[1] < self.delays + 1:
            raise ValueError("HankelDMD needs at least delays + 1 snapshots")
        embedded = delay_embedding(x, self.delays, backend=b.name, device=b.device)
        model = DMD(rank=self.rank, **self.dmd_options).fit(embedded)
        self.model_ = model
        self.backend_ = model.backend_
        self.n_features_in_, self.n_snapshots_ = x.shape
        self.rank_ = model.rank_
        self.eigenvalues_ = model.eigenvalues_
        self.amplitudes_ = model.amplitudes_
        self.embedded_modes_ = model.modes_
        self.modes_ = model.modes_[:self.n_features_in_]
        self._fitted = True
        return self

    def _check_fit(self):
        if not self._fitted:
            raise RuntimeError("Call fit before using this model")

    def reconstruct(self, steps=None):
        """Reconstruct ``steps`` physical snapshots from training time zero.

        ``None`` returns the training length. Unlike DMD, only a contiguous
        integer length is accepted, because overlapping windows are averaged.
        """
        self._check_fit()
        length = self.n_snapshots_ if steps is None else _positive_int(steps, "steps")
        if length < self.delays:
            raise ValueError("steps must be at least delays")
        width = length - self.delays + 1
        windows = self.model_.reconstruct(width)
        b = self.backend_
        result = b.zeros((self.n_features_in_, length), dtype=windows.dtype)
        counts = b.zeros((length,), dtype=windows.real.dtype)
        for j in range(self.delays):
            result[:, j:j + width] += windows[j * self.n_features_in_:(j + 1) * self.n_features_in_]
            counts[j:j + width] += 1
        return result / counts

    def predict(self, window):
        """Predict one physical sample from an observed ``(n, delays)`` window."""
        self._check_fit()
        x = _matrix(window, self.backend_, "window")
        if x.shape != (self.n_features_in_, self.delays):
            raise ValueError("window must have shape (n_features_in_, delays)")
        embedded = self.backend_.concat([x[:, j:j + 1] for j in range(self.delays)], axis=0)
        return self.model_.predict(embedded)[-self.n_features_in_:, 0]

    def forecast(self, n_steps):
        """Return ``n_steps`` samples immediately after the training sequence."""
        self._check_fit()
        return self.model_.forecast(n_steps)[-self.n_features_in_:]

    def diagnostics(self):
        self._check_fit()
        result = dict(self.model_.diagnostics())
        result.update(variant="hankel", delays=self.delays,
                      physical_features=self.n_features_in_, physical_snapshots=self.n_snapshots_)
        return result


class DMDc:
    """Dense controlled linear dynamics ``x[k+1] = A x[k] + B u[k]``.

    ``rank`` truncates the regression input, not the output state. This v0.1
    implementation stores dense ``A_`` and is intended for moderate state size.
    Provide ``B`` to fit when the control operator is known. Otherwise the
    concatenated state/control snapshots must provide sufficient excitation.
    """

    def __init__(self, rank=None, *, backend="auto", device=None, dt=1.0):
        if not isinstance(dt, Real) or isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.rank, self.backend, self.device, self.dt = rank, backend, device, float(dt)
        self._fitted = False

    def fit(self, X, U, Y=None, *, B=None):
        """Fit consecutive ``X`` or explicit snapshot pairs ``X, Y``.

        Controls ``U`` have shape ``(n_controls, n_pairs)``. With consecutive
        snapshots the number of controls is exactly ``X.shape[1] - 1``.
        ``B`` is an optional known ``(n_features, n_controls)`` operator.
        """
        self._fitted = False
        b = get_backend(X, backend=self.backend, device=self.device)
        data = _matrix(X, b, "X")
        controls = _matrix(U, b, "U")
        consecutive = Y is None
        if consecutive:
            if data.shape[1] < 2:
                raise ValueError("Consecutive X needs at least two snapshots")
            x, y = data[:, :-1], data[:, 1:]
        else:
            x, y = data, _matrix(Y, b, "Y")
            if y.shape != x.shape:
                raise ValueError("X and Y must have identical shapes")
        if controls.shape[1] != x.shape[1]:
            raise ValueError("U must have one column per transition pair")
        x, y, controls = _promote(b, x, y, controls)
        if B is None:
            omega = b.concat([x, controls], axis=0)
            inverse, r, numerical = _rank_pinv(omega, b, self.rank)
            operator = y @ inverse
            a, control = operator[:, :x.shape[0]], operator[:, x.shape[0]:]
        else:
            control = _matrix(B, b, "B")
            if control.shape != (x.shape[0], controls.shape[0]):
                raise ValueError("B must have shape (n_features, n_controls)")
            x, y, controls, control = _promote(b, x, y, controls, control)
            inverse, r, numerical = _rank_pinv(x, b, self.rank)
            a = (y - control @ controls) @ inverse
        self.backend_ = b
        copy = (lambda value: value.copy()) if b.name == "numpy" else (lambda value: value.clone())
        self.A_, self.B_, self.rank_ = a, copy(control), r
        self.n_features_in_, self.n_controls_ = x.shape[0], controls.shape[0]
        self.eigenvalues_, self.modes_ = b.eig(a)
        self._x0, self._last_snapshot = copy(x[:, 0]), copy(y[:, -1])
        self._controls = copy(controls)
        self._consecutive = consecutive
        self._n_pairs = x.shape[1]
        self.full_regression_rank_ = numerical
        self.required_regression_rank_ = x.shape[0] + (controls.shape[0] if B is None else 0)
        self.identifiable_ = r >= self.required_regression_rank_
        residual = a @ x + control @ controls - y
        self._relative_residual = b.norm(residual) / max(b.norm(y), np.finfo(float).tiny)
        self._fitted = True
        return self

    def _check_fit(self):
        if not self._fitted:
            raise RuntimeError("Call fit before using this model")

    def predict(self, X, U):
        """Predict one step per state/control column; vectors are accepted."""
        self._check_fit()
        b = self.backend_
        x, u = b.asarray(X), b.asarray(U)
        vector = x.ndim == 1
        if vector:
            x = x[:, None]
        if u.ndim == 1:
            u = u[:, None]
        x, u = _matrix(x, b, "X"), _matrix(u, b, "U")
        if x.shape[0] != self.n_features_in_ or u.shape != (self.n_controls_, x.shape[1]):
            raise ValueError("X and U must have matching columns and fitted feature/control dimensions")
        a, control, x, u = _promote(b, self.A_, self.B_, x, u)
        y = a @ x + control @ u
        return y[:, 0] if vector else y

    def simulate(self, x0, U):
        """Return initial state plus one state per supplied control column."""
        self._check_fit()
        b = self.backend_
        u = _matrix(U, b, "U")
        state = b.asarray(x0)
        if state.ndim != 1 or state.shape[0] != self.n_features_in_:
            raise ValueError("x0 must be a vector of length n_features_in_")
        if not bool(b.xp.isfinite(state).all()):
            raise ValueError("x0 must contain finite values")
        if u.shape[0] != self.n_controls_:
            raise ValueError("U has the wrong number of controls")
        a, control, state, u = _promote(b, self.A_, self.B_, state, u)
        states = [state[:, None]]
        for k in range(u.shape[1]):
            state = a @ state + control @ u[:, k]
            states.append(state[:, None])
        return b.concat(states, axis=1)

    def reconstruct(self):
        """Simulate the training controls; requires consecutive training data."""
        self._check_fit()
        if not self._consecutive:
            raise ValueError("Explicit unordered X/Y pairs cannot be reconstructed as a trajectory")
        return self.simulate(self._x0, self._controls)

    def forecast(self, U):
        """Forecast after the last training target using supplied future controls."""
        self._check_fit()
        return self.simulate(self._last_snapshot, U)[:, 1:]

    def diagnostics(self):
        self._check_fit()
        notes = []
        if self.full_regression_rank_ < self.required_regression_rank_:
            notes.append("Insufficient regression input rank: state/control excitation does not identify unique full operators.")
        if self.rank_ < self.full_regression_rank_:
            notes.append("User rank truncation discards resolved regression directions; full operators are not identified by this fit.")
        if not self.identifiable_:
            notes.append("A small training residual does not establish operator identifiability.")
        return {"variant": "dmdc", "backend": self.backend_.name,
                "device": str(self.backend_.device), "rank": self.rank_,
                "features": self.n_features_in_, "controls": self.n_controls_,
                "snapshot_pairs": self._n_pairs, "relative_residual": self._relative_residual,
                "full_regression_rank": self.full_regression_rank_,
                "required_regression_rank": self.required_regression_rank_,
                "identifiable": self.identifiable_, "notes": notes,
                "dense_state_operator": True}


class RobustDMD(DMD):
    """DMD after principal component pursuit (low rank + sparse corruption).

    This is a two-stage RPCA/DMD pipeline, not a new robust DMD estimator.
    RPCA assumes incoherent low-rank structure and sparse corruption; its
    output is not guaranteed to improve forecasting under dense noise.
    For float32/complex64 inputs, use ``rpca_tol=1e-5``: the default strict
    tolerance may fall below the achievable precision and issue a warning.
    """

    def __init__(self, rank=None, *, rpca_lambda=None, rpca_tol=1e-7,
                 rpca_max_iter=1000, **dmd_options):
        super().__init__(rank=rank, **dmd_options)
        if rpca_lambda is not None and (not isinstance(rpca_lambda, Real)
                or isinstance(rpca_lambda, bool) or not math.isfinite(rpca_lambda) or rpca_lambda <= 0):
            raise ValueError("rpca_lambda must be finite and positive")
        if not isinstance(rpca_tol, Real) or isinstance(rpca_tol, bool) or not math.isfinite(rpca_tol) or rpca_tol <= 0:
            raise ValueError("rpca_tol must be finite and positive")
        self.rpca_lambda = rpca_lambda
        self.rpca_tol = float(rpca_tol)
        self.rpca_max_iter = _positive_int(rpca_max_iter, "rpca_max_iter")

    def fit(self, X, Y=None):
        self._fitted = False
        if Y is not None:
            raise ValueError("RobustDMD accepts a consecutive snapshot sequence X only")
        b = get_backend(X, backend=self.backend, device=self.device)
        data = _matrix(X, b, "X")
        if data.shape[1] < 2:
            raise ValueError("X needs at least two snapshots")
        lam = self.rpca_lambda or 1 / math.sqrt(max(data.shape))
        tiny = (np.finfo(data.real.dtype).tiny if b.name == "numpy"
                else b.xp.finfo(data.real.dtype).tiny)
        _, singular, _ = b.svd(data)
        spectral = float(b.scalar(singular[0]))
        norm_data = b.norm(data)
        if spectral == 0:
            raise ValueError("X has numerical rank zero")
        dual_norm = max(spectral, float(b.scalar(abs(data).max())) / lam)
        dual = data / dual_norm
        low = b.zeros(data.shape, dtype=data.dtype)
        sparse = b.zeros(data.shape, dtype=data.dtype)
        mu, mu_max = 1.25 / spectral, 1.25e7 / spectral
        self.rpca_converged_ = False
        for iteration in range(1, self.rpca_max_iter + 1):
            previous_sparse = sparse
            u, s, vh = b.svd(data - sparse + dual / mu)
            shrunk = (s - 1 / mu).clip(0)
            low = (u * shrunk) @ vh
            value = data - low + dual / mu
            magnitude = abs(value)
            # Clamp avoids undefined complex phase at zero for both backends.
            thresholded = (magnitude - lam / mu).clip(0)
            sparse = (value / magnitude.clip(tiny)) * thresholded
            residual = data - low - sparse
            dual = dual + mu * residual
            relative = b.norm(residual) / norm_data
            dual_error = mu * b.norm(sparse - previous_sparse) / max(1.0, b.norm(dual))
            error = max(relative, dual_error)
            if error <= self.rpca_tol:
                self.rpca_converged_ = True
                break
            if relative > dual_error:
                mu = min(mu * 1.5, mu_max)
        self.low_rank_, self.sparse_ = low, sparse
        self.rpca_iterations_, self.rpca_residual_ = iteration, relative
        self.rpca_dual_error_, self.rpca_error_ = dual_error, error
        self.rpca_lambda_ = lam
        if not self.rpca_converged_:
            warnings.warn("RPCA did not converge within rpca_max_iter; inspect rpca_error_",
                          RuntimeWarning, stacklevel=2)
        return super().fit(low)

    def diagnostics(self):
        result = dict(super().diagnostics())
        result.update(variant="rpca_dmd", rpca_converged=self.rpca_converged_,
                      rpca_iterations=self.rpca_iterations_, rpca_residual=self.rpca_residual_,
                      rpca_dual_error=self.rpca_dual_error_, rpca_error=self.rpca_error_,
                      rpca_lambda=self.rpca_lambda_)
        return result
