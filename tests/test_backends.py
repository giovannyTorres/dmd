import numpy as np
import pytest

from modaldyn._backend import Backend, get_backend
from modaldyn.core import DMD, FbDMD, RandomizedDMD, TLSqDMD


def torch_or_skip():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("factory", [DMD, TLSqDMD, FbDMD, RandomizedDMD])
def test_torch_cpu_parity(dtype, factory):
    torch = torch_or_skip()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2, 20)).astype(dtype)
    A = np.array([[.9, -.1], [.1, .9]], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        X = X + 1j * rng.normal(size=X.shape).astype(dtype)
        A[0, 0] += .04j
    Y = A @ X
    cpu = factory(rank=2).fit(X, Y)
    model = factory(rank=2, backend="torch", device="cpu").fit(X, Y)
    assert model.modes_.device.type == "cpu"
    assert not model.modes_.requires_grad
    tol = 1e-5 if X.dtype in (np.float32, np.complex64) else 1e-11
    np.testing.assert_allclose(model.predict(X).numpy(), cpu.predict(X), atol=tol, rtol=tol)
    np.testing.assert_allclose(model.reconstruct(5).numpy(), cpu.reconstruct(5), atol=tol, rtol=tol)
    np.testing.assert_allclose(model.forecast(3).numpy(), cpu.forecast(3), atol=tol, rtol=tol)
    auto = DMD(rank=2).fit(torch.as_tensor(X), torch.as_tensor(Y))
    assert auto.backend_.name == "torch"


def test_torch_zero_eigenvalue_time_zero():
    torch_or_skip()
    m = DMD(backend="torch").fit(np.eye(2), np.zeros((2, 2)))
    np.testing.assert_allclose(m.reconstruct(3).numpy(), [[1., 0., 0.], [0., 0., 0.]], atol=1e-14)
    assert m.forecast(0).shape == (2, 0)
    assert m.reconstruct(0).shape == (2, 0)


def test_explicit_numpy_accepts_conjugated_cpu_tensor():
    torch = torch_or_skip()
    x = torch.tensor([[1+1j, 2+2j, 4+4j]], dtype=torch.complex128).conj()
    model = DMD(backend="numpy").fit(x)
    np.testing.assert_allclose(model.forecast(1), [[8-8j]], atol=1e-12)


def test_explicit_device_and_backend_validation():
    with pytest.raises(ValueError, match="NumPy"):
        get_backend(np.eye(2), backend="numpy", device="cuda")
    with pytest.raises(ValueError, match="backend"):
        Backend("unknown")
    torch = torch_or_skip()
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            DMD(device="cuda").fit(np.eye(2))
        with pytest.raises(RuntimeError, match="CUDA"):
            DMD(backend="torch", device="cuda").fit(np.eye(2))
    with pytest.raises(ValueError, match="Supported"):
        Backend("torch", device="meta")


def test_integer_half_precision_and_detached_inputs():
    torch = torch_or_skip()
    for backend in ["numpy", "torch"]:
        m = DMD(backend=backend).fit(np.array([[4, 2, 1]]))
        np.testing.assert_allclose(m.backend_.to_numpy(m.forecast(2)), [[.5, .25]], atol=1e-12)
    a = torch.tensor([[4., 2., 1.]], requires_grad=True, dtype=torch.float64)
    m = DMD().fit(a)
    assert not m.modes_.requires_grad
    np.testing.assert_allclose(m.forecast(2).numpy(), [[.5, .25]], atol=1e-12)
    m = DMD().fit(a.detach().to(torch.float16))
    assert m._training_X.dtype == torch.float32


@pytest.mark.cuda
@pytest.mark.parametrize("factory", [DMD, TLSqDMD, FbDMD, RandomizedDMD])
def test_cuda_device_and_numerical_parity(factory):
    torch = torch_or_skip()
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware not available; GPU numerical checks not executed")
    rng = np.random.default_rng(13)
    X = rng.normal(size=(2, 30))
    A = np.array([[.97, -.1], [.1, .97]])
    Y = A @ X
    cpu = factory(rank=2).fit(X, Y)
    model = factory(rank=2, backend="torch", device="cuda").fit(X, Y)
    assert model.modes_.is_cuda
    assert model.eigenvalues_.is_cuda
    assert model.forecast(3).is_cuda
    np.testing.assert_allclose(model.predict(X).cpu().numpy(), cpu.predict(X), atol=2e-10)
    np.testing.assert_allclose(model.forecast(3).cpu().numpy(), cpu.forecast(3), atol=2e-10)
    with pytest.raises(ValueError, match="transfer"):
        DMD(backend="numpy").fit(torch.as_tensor(X, device="cuda"))
