"""Small, explicit NumPy/PyTorch linear algebra boundary (no silent fallback)."""
from __future__ import annotations

import importlib
import numpy as np


class Backend:
    def __init__(self, name="numpy", device=None):
        self.name = name
        if name == "numpy":
            if device not in (None, "cpu"):
                raise ValueError("NumPy only supports device='cpu'; use backend='torch' for CUDA")
            self.xp = np
            self.device = "cpu"
        elif name == "torch":
            try:
                self.xp = importlib.import_module("torch")
            except ImportError as e:
                raise ImportError("Install modaldyn[torch] and a PyTorch build for your accelerator") from e
            self.device = str(self.xp.device(device or "cpu"))
            if self.device.startswith("cuda") and not self.xp.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; install CUDA-enabled PyTorch and check the driver")
            if self.device.split(":")[0] not in ("cpu", "cuda"):
                raise ValueError("Supported torch devices: cpu and cuda")
        else:
            raise ValueError("backend must be 'auto', 'numpy', or 'torch'")

    def asarray(self, x):
        if self.name == "numpy":
            if type(x).__module__.startswith("torch"):
                if x.device.type != "cpu":
                    raise ValueError("Explicitly transfer GPU input to CPU before using NumPy")
                x = x.detach().resolve_conj().resolve_neg().numpy()
            a = np.asarray(x)
            if a.dtype.kind not in "fciub":
                raise TypeError("Expected real or complex numeric data")
            if a.dtype.kind in "iub":
                a = a.astype(np.float64)
            if a.dtype in (np.float16,):
                a = a.astype(np.float32)
            return a
        t = self.xp
        a = t.as_tensor(x, device=self.device).detach()
        if not (a.is_floating_point() or a.is_complex()):
            a = a.to(t.float64)
        if a.dtype in (t.float16, t.bfloat16):
            a = a.to(t.float32)
        return a

    def svd(self, x):
        return self.xp.linalg.svd(x, full_matrices=False)

    def eig(self, x):
        return self.xp.linalg.eig(x)

    def _promote(self, a, b):
        if self.name == "torch":
            dtype = self.xp.promote_types(a.dtype, b.dtype)
            return a.to(dtype), b.to(dtype)
        dtype = np.result_type(a.dtype, b.dtype)
        return a.astype(dtype, copy=False), b.astype(dtype, copy=False)

    def matmul(self, a, b):
        a, b = self._promote(a, b)
        return a @ b

    def lstsq(self, a, b):
        a, b = self._promote(a, b)
        # pinv uses SVD on both devices, including rank-deficient matrices.
        return self.matmul(self.pinv(a), b)

    def pinv(self, x, rcond=None):
        if rcond is None:
            # Make both engines use dtype- and shape-aware numerical rank.
            eps = self.xp.finfo(x.real.dtype).eps
            rcond = max(x.shape) * eps
        return self.xp.linalg.pinv(x, rcond=rcond)

    def qr(self, x):
        return self.xp.linalg.qr(x, mode="reduced")

    def to_numpy(self, x):
        return x.detach().cpu().numpy() if self.name == "torch" else np.asarray(x)

    def scalar(self, x):
        return x.item() if hasattr(x, "item") else x

    def norm(self, x):
        size = x.size if self.name == "numpy" else x.numel()
        scale = float(self.scalar(abs(x).max())) if size else 0.0
        if scale == 0:
            return 0.0
        return scale * float(self.scalar(self.xp.linalg.norm(x / scale)))

    def concat(self, arrays, axis=0):
        return self.xp.concatenate(arrays, axis=axis) if self.name == "numpy" else self.xp.cat(arrays, dim=axis)

    def zeros(self, shape, dtype=None):
        return self.xp.zeros(shape, dtype=dtype) if self.name == "numpy" else self.xp.zeros(shape, dtype=dtype, device=self.device)

    def eye(self, n, dtype=None):
        return self.xp.eye(n, dtype=dtype) if self.name == "numpy" else self.xp.eye(n, dtype=dtype, device=self.device)

    def sqrt(self, x):
        return self.xp.sqrt(x)

    def random_normal(self, shape, dtype, seed=0):
        # Generate on CPU with a local RNG for engine-independent random sketches.
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(shape)
        if "complex" in str(dtype):
            z = (z + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        a = self.asarray(z)
        return a.astype(dtype) if self.name == "numpy" else a.to(dtype)


def get_backend(x=None, backend="auto", device=None):
    if backend == "auto":
        is_torch = type(x).__module__.startswith("torch")
        backend = "torch" if is_torch or (device and str(device).startswith("cuda")) else "numpy"
        if is_torch and device is None:
            device = str(x.device)
    elif backend == "torch" and device is None and type(x).__module__.startswith("torch"):
        device = str(x.device)
    return Backend(backend, device)


def validate_matrix(x, b, name="X", min_columns=1):
    a = b.asarray(x)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < min_columns:
        raise ValueError(f"{name} must be (features, snapshots) with >= {min_columns} columns")
    if not bool(b.scalar(b.xp.isfinite(a).all())):
        raise ValueError(f"{name} contains NaN or infinity")
    return a
