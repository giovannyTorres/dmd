"""Dynamic mode decomposition with explicit numerical and device semantics."""
from __future__ import annotations

import numbers
import warnings
import numpy as np

from ._backend import get_backend, validate_matrix


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class DMD:
    """Exact DMD. Columns are snapshots; ``dt`` is the uniform sampling interval.

    ``rank=None`` retains numerical rank, an integer caps it, and a float in
    (0, 1] selects squared-singular-value energy. No automatic device fallback.
    TLSQ, forward/backward and randomized SVD are opt-in, combinable algorithms.
    """

    def __init__(self, rank=None, *, backend="auto", device=None, dt=1.0,
                 tlsq_rank=0, forward_backward=False, randomized=False,
                 oversampling=10, power_iters=2, seed=0, rcond=None):
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, numbers.Real):
                raise ValueError("rank must be None, a positive integer, or energy in (0,1]")
            if isinstance(rank, numbers.Integral):
                _integer(rank, "rank", 1)
            elif not 0 < rank <= 1:
                raise ValueError("energy rank must lie in (0,1]")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if rcond is not None and (not np.isfinite(rcond) or not 0 <= rcond < 1):
            raise ValueError("rcond must lie in [0,1)")
        self.rank, self.backend, self.device, self.dt = rank, backend, device, float(dt)
        self.tlsq_rank = _integer(tlsq_rank, "tlsq_rank")
        self.oversampling = _integer(oversampling, "oversampling")
        self.power_iters = _integer(power_iters, "power_iters")
        self.seed = _integer(seed, "seed")
        self.forward_backward, self.randomized, self.rcond = forward_backward, randomized, rcond
        if randomized and (not isinstance(rank, numbers.Integral) or isinstance(rank, bool)):
            raise ValueError("Randomized DMD requires an explicit integer rank")
        self._fitted = False

    def _svd(self, X):
        b = self.backend_
        if not self.randomized:
            return b.svd(X)
        width = min(min(X.shape), self.rank + self.oversampling)
        Q, _ = b.qr(b.matmul(X, b.random_normal((X.shape[1], width), X.dtype, self.seed)))
        for _ in range(self.power_iters):
            Z, _ = b.qr(b.matmul(X.conj().T, Q))
            Q, _ = b.qr(b.matmul(X, Z))
        U, s, Vh = b.svd(b.matmul(Q.conj().T, X))
        return b.matmul(Q, U), s, Vh

    def fit(self, X, Y=None):
        self._fitted = False
        b = self.backend_ = get_backend(X, self.backend, self.device)
        data = validate_matrix(X, b, min_columns=2 if Y is None else 1)
        if Y is None:
            X, Y = data[:, :-1], data[:, 1:]
            count = data.shape[1]
        else:
            X, Y = data, validate_matrix(Y, b, "Y")
            if X.shape != Y.shape:
                raise ValueError("X and Y must have identical shapes")
            count = X.shape[1] + 1
        X, Y = b._promote(X, Y)
        # Own the fitted state: mutation of caller arrays cannot alter forecasts.
        copy = (lambda a: a.copy()) if b.name == "numpy" else (lambda a: a.clone())
        self._training_X, self._training_Y = copy(X), copy(Y)
        self._x0, self._last_snapshot = copy(X[:, 0]), copy(Y[:, -1])
        self.n_features_in_, self.n_snapshots_ = X.shape[0], count
        if self.tlsq_rank:
            Z = b.concat((X, Y))
            _, _, Vh_tls = b.svd(Z)
            q = min(self.tlsq_rank, min(Z.shape))
            V = Vh_tls[:q].conj().T
            X = b.matmul(b.matmul(X, V), V.conj().T)
            Y = b.matmul(b.matmul(Y, V), V.conj().T)
        U, s, Vh = self._svd(X)
        eps = b.xp.finfo(s.dtype).eps
        tol = (self.rcond if self.rcond is not None else max(X.shape) * eps) * float(b.scalar(s[0]))
        numerical_rank = int(b.scalar((s > tol).sum()))
        if numerical_rank == 0:
            raise ValueError("X has zero numerical rank; dynamics cannot be identified")
        if self.rank is None:
            rank = numerical_rank
        elif isinstance(self.rank, numbers.Integral):
            rank = min(int(self.rank), numerical_rank)
        else:
            # Normalize before summing to avoid overflow on large magnitudes.
            energy = (b.to_numpy(s[:numerical_rank]) / float(b.scalar(s[0]))) ** 2
            rank = min(int(np.searchsorted(np.cumsum(energy) / energy.sum(), self.rank)) + 1, numerical_rank)
        self.rank_, self.numerical_rank_ = rank, numerical_rank
        self.singular_values_ = s
        U, s, Vh = U[:, :rank], s[:rank], Vh[:rank]
        lift = b.matmul(Y, Vh.conj().T) / s[None, :]
        A = b.matmul(U.conj().T, lift)
        if self.forward_backward:
            # Same spatial basis in both directions; root of Af Ab^{-1}.
            Xr, Yr = b.matmul(U.conj().T, X), b.matmul(U.conj().T, Y)
            Ab = b.matmul(Xr, b.pinv(Yr, self.rcond))
            if b.norm(Ab) == 0 or int(b.scalar((b.svd(Ab)[1] > max(Ab.shape) * eps * b.svd(Ab)[1][0]).sum())) < rank:
                raise ValueError("Backward operator is singular; forward/backward DMD is undefined")
            C = b.matmul(A, b.pinv(Ab, self.rcond))
            values, W = b.eig(C)
            # eig returns complex on torch; NumPy needs explicit complex root.
            if b.name == "numpy":
                values = values.astype(complex)
            A = b.matmul(b.matmul(W, b.xp.diag(b.sqrt(values))), b.pinv(W, self.rcond))
        values, W = b.eig(A)
        modes = b.matmul(U if self.forward_backward else lift, W)
        # Exact lifting loses eigenvectors of zero eigenvalues. Projected vectors
        # supply those directions so initial conditions can still be represented.
        for j in range(rank):
            if b.norm(modes[:, j]) <= eps * max(1.0, b.norm(lift)):
                modes[:, j] = b.matmul(U, W)[:, j]
        self.eigenvalues_, self.modes_ = values, modes
        self.amplitudes_ = b.lstsq(modes, self._x0)
        self._fitted = True
        return self

    def _check(self):
        if not self._fitted:
            raise RuntimeError("Call fit before using the model")

    def predict(self, X):
        """Apply the fitted spectral operator to one vector or snapshot matrix."""
        self._check()
        b = self.backend_
        X = b.asarray(X)
        if X.ndim not in (1, 2) or X.shape[0] != self.n_features_in_:
            raise ValueError("X must have the fitted feature dimension")
        if not bool(b.scalar(b.xp.isfinite(X).all())):
            raise ValueError("X contains NaN or infinity")
        coeff = b.lstsq(self.modes_, X)
        scale = self.eigenvalues_ if X.ndim == 1 else self.eigenvalues_[:, None]
        return b.matmul(self.modes_, scale * coeff)

    def _steps(self, steps):
        if steps is None:
            return np.arange(self.n_snapshots_)
        if isinstance(steps, numbers.Integral) and not isinstance(steps, bool):
            return np.arange(_integer(steps, "steps"))
        a = np.asarray(steps)
        if a.ndim != 1 or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("steps must be a nonnegative length or a vector of nonnegative integer indices")
        return a

    def _evolve(self, x0, steps):
        b = self.backend_
        k = b.asarray(steps)
        amplitudes = b.lstsq(self.modes_, x0)
        powers = self.eigenvalues_[:, None] ** k[None, :]
        # PyTorch's complex pow evaluates 0**0 as NaN. The discrete propagator
        # at time zero is the identity, including zero eigenvalues.
        powers[:, k == 0] = 1
        return b.matmul(self.modes_, amplitudes[:, None] * powers)

    def reconstruct(self, steps=None):
        """Modal reconstruction from the first X column, including index zero.

        For explicitly paired X,Y, columns need not be a trajectory; the default
        length is pairs+1, and this rollout still starts at the first X column.
        """
        self._check()
        return self._evolve(self._x0, self._steps(steps))

    def forecast(self, n_steps):
        """Return future snapshots after the final training Y column."""
        self._check()
        n_steps = _integer(n_steps, "n_steps")
        return self._evolve(self._last_snapshot, np.arange(1, n_steps + 1))

    @property
    def frequencies_(self):
        self._check()
        return self.backend_.xp.angle(self.eigenvalues_) / (2 * np.pi * self.dt)

    @property
    def growth_rates_(self):
        self._check()
        b = self.backend_
        with np.errstate(divide="ignore"):
            return b.xp.log(b.xp.abs(self.eigenvalues_)) / self.dt

    def diagnostics(self):
        """JSON-safe scalars; reductions synchronize CUDA, arrays stay on device."""
        self._check()
        b = self.backend_
        target = self._training_Y
        denom = b.norm(target)
        error = b.norm(self.predict(self._training_X) - target)
        if self.rank_ < self.numerical_rank_:
            notes = ["Rank truncation discards resolved singular directions."]
        else:
            notes = []
        if self.randomized:
            notes.append("Numerical rank is measured in the randomized sketch; omitted directions may remain undetected.")
        if self.forward_backward:
            notes.append("The principal matrix square root has branch ambiguity; signs and high-frequency modes may differ from forward DMD.")
        radii = b.xp.abs(self.eigenvalues_)
        radius = float(b.scalar(radii.max()))
        if radius > 1 + 100 * b.xp.finfo(radii.dtype).eps:
            notes.append("Eigenvalues outside the unit circle imply growing modal forecasts.")
        if b.norm(self.modes_) == 0:
            warnings.warn("All modes are zero", RuntimeWarning, stacklevel=2)
        mode_s = b.svd(self.modes_)[1]
        smallest_mode_s = float(b.scalar(mode_s[-1]))
        mode_condition = float(b.scalar(mode_s[0])) / smallest_mode_s if smallest_mode_s else np.inf
        if not np.isfinite(mode_condition) or mode_condition > 1 / np.sqrt(b.xp.finfo(mode_s.dtype).eps):
            notes.append("Modes are ill-conditioned; a diagonal modal expansion may not represent defective dynamics reliably.")
        initial_norm = b.norm(self._x0)
        initial_error = b.norm(b.matmul(self.modes_, self.amplitudes_) - self._x0)
        singular = self.singular_values_
        return {"variant": type(self).__name__, "backend": b.name, "device": b.device,
                "dtype": str(self._training_X.dtype), "rank": self.rank_,
                "numerical_rank": self.numerical_rank_, "features": self.n_features_in_,
                "numerical_rank_scope": "randomized_sketch" if self.randomized else "snapshot_matrix",
                "snapshot_pairs": int(target.shape[1]), "dt": self.dt,
                "relative_one_step_error": error / denom if denom else None,
                "retained_condition_number": float(b.scalar(singular[0] / singular[self.rank_ - 1])),
                "mode_condition_number": mode_condition if np.isfinite(mode_condition) else None,
                "relative_initial_projection_error": initial_error / initial_norm if initial_norm else None,
                "spectral_radius": radius, "tlsq_rank": self.tlsq_rank,
                "forward_backward": bool(self.forward_backward), "randomized": bool(self.randomized),
                "notes": notes}


class TLSqDMD(DMD):
    """Total-least-squares preprocessing of paired snapshots before exact DMD."""
    def __init__(self, rank=None, *, tlsq_rank=2, **kwargs):
        super().__init__(rank, tlsq_rank=tlsq_rank, **kwargs)


class FbDMD(DMD):
    """Forward/backward DMD with the principal matrix square root convention."""
    def __init__(self, rank=None, **kwargs):
        super().__init__(rank, forward_backward=True, **kwargs)


class RandomizedDMD(DMD):
    """Seeded randomized range finder followed by reduced exact DMD."""
    def __init__(self, rank, **kwargs):
        super().__init__(rank, randomized=True, **kwargs)
