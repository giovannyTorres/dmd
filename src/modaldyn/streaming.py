"""DMD from sufficient statistics without retaining snapshot history."""

from __future__ import annotations

import numbers

import numpy as np

from ._backend import get_backend


def _matmul(backend, left, right):
    """Promote mixed real/complex operands consistently on both backends."""
    if backend.name == "torch":
        dtype = backend.xp.promote_types(left.dtype, right.dtype)
        left = left.to(dtype=dtype)
        right = right.to(dtype=dtype)
    return left @ right


class StreamingDMD:
    """Accumulate a least-squares DMD operator on NumPy or PyTorch.

    Snapshots are columns. ``partial_fit(X, Y)`` accumulates paired columns;
    ``partial_fit(X)`` uses only transitions *inside* that batch. Consecutive
    calls are not joined automatically, so callers can safely mix trajectories.

    The retained state is two ``(n_features, n_features)`` matrices. This avoids
    storing history, but is not a low-memory method for very large state spaces.
    ``forgetting_factor`` discounts existing sufficient statistics once per
    accepted batch, independently of the number of pairs in that batch.

    ``rank`` truncates the pseudoinverse of the Gram matrix. It does not change
    storage complexity. Eigenvectors are computed from the complete fitted
    operator; ``modes_`` therefore has ``n_features`` columns, including null
    modes when the operator is rank deficient. Forecasts use the operator
    directly and do not require an invertible modal basis.
    """

    def __init__(
        self,
        rank=None,
        *,
        backend="auto",
        device=None,
        dt=1.0,
        forgetting_factor=1.0,
        rcond=None,
    ):
        if rank is not None and (
            isinstance(rank, bool) or not isinstance(rank, numbers.Integral) or rank < 1
        ):
            raise ValueError("rank must be None or a positive integer")
        if not isinstance(dt, numbers.Real) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if (
            not isinstance(forgetting_factor, numbers.Real)
            or not np.isfinite(forgetting_factor)
            or not 0 < forgetting_factor <= 1
        ):
            raise ValueError("forgetting_factor must be in (0, 1]")
        if rcond is not None and (
            not isinstance(rcond, numbers.Real) or not np.isfinite(rcond) or rcond < 0
        ):
            raise ValueError("rcond must be None or finite and nonnegative")
        if backend not in {"auto", "numpy", "torch"}:
            raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
        self.rank = rank
        self.backend = backend
        self.device = device
        self.dt = float(dt)
        self.forgetting_factor = float(forgetting_factor)
        self.rcond = rcond
        self.n_batches_ = 0
        self.n_pairs_seen_ = 0
        self.effective_pairs_ = 0.0
        self._fitted = False

    def _validate_pair(self, X, Y):
        b = getattr(self, "backend_", None)
        if b is None:
            b = get_backend(X, backend=self.backend, device=self.device)
        x = b.asarray(X)
        if x.ndim != 2 or min(x.shape) < 1:
            raise ValueError("X must be a nonempty 2-D array of column snapshots")
        if Y is None:
            if x.shape[1] < 2:
                raise ValueError("X needs at least two columns when Y is omitted")
            y = x[:, 1:]
            x = x[:, :-1]
        else:
            y = b.asarray(Y)
            if y.ndim != 2 or y.shape != x.shape:
                raise ValueError("X and Y must have the same nonempty 2-D shape")
        if not bool(b.xp.isfinite(x).all()) or not bool(b.xp.isfinite(y).all()):
            raise ValueError("snapshots must contain only finite values")
        if hasattr(self, "n_features_in_") and x.shape[0] != self.n_features_in_:
            raise ValueError("feature count must remain constant between batches")
        if self.rank is not None and self.rank > x.shape[0]:
            raise ValueError("rank cannot exceed the number of features")
        return b, x, y

    def partial_fit(self, X, Y=None):
        """Accumulate one batch and invalidate previously finalized outputs.

        All validation precedes state mutation. Inputs are not retained, and
        editing an input array after the call does not change the model.
        Call :meth:`finalize` after the last batch before prediction.
        """
        b, x, y = self._validate_pair(X, Y)
        gram = _matmul(b, x, x.conj().T)
        cross = _matmul(b, y, x.conj().T)
        if self.n_batches_:
            gram = self.forgetting_factor * self._gram + gram
            cross = self.forgetting_factor * self._cross + cross
        if not bool(b.xp.isfinite(gram).all()) or not bool(b.xp.isfinite(cross).all()):
            raise ValueError("sufficient statistics overflowed; rescale the snapshots")
        self.backend_ = b
        self._gram = gram
        self._cross = cross
        self.n_features_in_ = x.shape[0]
        self.n_pairs_seen_ += x.shape[1]
        self.effective_pairs_ = (
            self.forgetting_factor * self.effective_pairs_ + x.shape[1]
        )
        self.n_batches_ += 1
        self._fitted = False
        for attr in ("operator_", "eigenvalues_", "modes_", "rank_", "gram_singular_values_"):
            self.__dict__.pop(attr, None)
        return self

    def finalize(self):
        """Solve ``A = H G_r^dagger`` from the accumulated statistics.

        Forming a Gram matrix squares the condition number of the snapshot
        matrix. Use batch DMD when conditioning is the primary concern.
        """
        if not self.n_batches_:
            raise RuntimeError("call partial_fit before finalize")
        b = self.backend_
        u, s, vh = b.svd(self._gram)
        threshold = (
            self.rcond if self.rcond is not None else self.n_features_in_ * b.xp.finfo(s.dtype).eps
        ) * float(b.scalar(s[0]))
        numerical_rank = int(b.scalar((s > threshold).sum()))
        rank = numerical_rank if self.rank is None else min(self.rank, numerical_rank)
        if rank == 0:
            raise ValueError("snapshots have zero numerical rank")
        # Use both SVD factors instead of assuming perfectly Hermitian data.
        gram_inverse = (vh[:rank].conj().T / s[:rank]) @ u[:, :rank].conj().T
        operator = _matmul(b, self._cross, gram_inverse)
        eigenvalues, modes = b.eig(operator)
        self.operator_ = operator
        self.eigenvalues_ = eigenvalues
        self.modes_ = modes
        self.rank_ = rank
        self.gram_singular_values_ = s
        self._fitted = True
        return self

    def fit(self, X, Y=None):
        """Reset, accumulate a batch, and finalize; failed fits preserve state."""
        fresh = type(self)(
            rank=self.rank,
            backend=self.backend,
            device=self.device,
            dt=self.dt,
            forgetting_factor=self.forgetting_factor,
            rcond=self.rcond,
        )
        fresh.partial_fit(X, Y).finalize()
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("call finalize after accumulating snapshots")

    def _state(self, X):
        self._check_fitted()
        b = self.backend_
        x = b.asarray(X)
        if x.ndim not in (1, 2) or x.shape[0] != self.n_features_in_:
            raise ValueError("state must have shape (n_features,) or (n_features, n_states)")
        if x.ndim == 2 and x.shape[1] == 0:
            raise ValueError("state matrix cannot be empty")
        if not bool(b.xp.isfinite(x).all()):
            raise ValueError("state must contain only finite values")
        return x

    def predict(self, X):
        """Advance one step, preserving vector/matrix shape and fitted backend."""
        x = self._state(X)
        return _matmul(self.backend_, self.operator_, x)

    def forecast(self, x0, n_steps):
        """Return future states ``[A x0, ..., A**n_steps x0]`` as columns.

        The initial state is excluded. ``x0`` must be one state, either a vector
        or a single-column matrix. An empty horizon returns shape ``(n, 0)``.
        """
        x = self._state(x0)
        if isinstance(n_steps, bool) or not isinstance(n_steps, numbers.Integral) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer")
        if x.ndim == 2:
            if x.shape[1] != 1:
                raise ValueError("forecast needs exactly one initial state")
            x = x[:, 0]
        if n_steps == 0:
            return self.backend_.zeros((self.n_features_in_, 0), dtype=x.dtype)
        future = []
        for _ in range(n_steps):
            x = _matmul(self.backend_, self.operator_, x)
            future.append(x[:, None])
        return self.backend_.concat(future, axis=1)

    def diagnostics(self):
        """Return JSON-compatible operating and conditioning metadata."""
        self._check_fitted()
        b = self.backend_
        s = self.gram_singular_values_
        condition = float(b.scalar(s[0] / s[self.rank_ - 1]))
        return {
            "variant": "StreamingDMD",
            "backend": b.name,
            "device": str(b.device),
            "n_features": self.n_features_in_,
            "n_batches": self.n_batches_,
            "n_pairs_seen": self.n_pairs_seen_,
            "effective_pairs": self.effective_pairs_,
            "rank": self.rank_,
            "dt": self.dt,
            "forgetting_factor": self.forgetting_factor,
            "gram_condition_number": condition,
            "retained_statistic_elements": 2 * self.n_features_in_**2,
            "boundary_transitions": "not inferred",
        }
