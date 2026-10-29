# modaldyn

A compact Dynamic Mode Decomposition toolkit with a NumPy CPU core and an
optional PyTorch CPU/CUDA backend. It uses snapshot matrices of shape
**`(features, snapshots)`** and makes device choice explicit.

Initial research release, version 0.1.0. CUDA execution is implemented; a passing
CPU suite or skipped CUDA tests do **not** establish validation on GPU hardware.
See [limitations](docs/limitations.md) and the actual CI/benchmark results for
the tested configurations. [Documentación en español](README.es.md).

## Install

Install the tagged source release:

```bash
python -m pip install 'git+https://github.com/giovannyTorres/dmd.git@v0.1.0'
```

Or from a cloned checkout:

```bash
python -m pip install .
```

Or install a built wheel from the release assets:

```bash
python -m pip install modaldyn-0.1.0-py3-none-any.whl
```

Python 3.10+ and NumPy 1.24+ are required. PyTorch is optional. Install the
appropriate [PyTorch build](https://pytorch.org/get-started/locally/) for your
CPU or NVIDIA CUDA system, then install `.[torch]`. Installing this extra alone
does not configure a CUDA driver. PyPI publication requires the separate
[release setup](docs/releasing.md); a source release is not proof of a PyPI upload.

## Fit, reconstruct and forecast

```python
import numpy as np
from modaldyn import DMD

t = np.arange(160) * 0.1
snapshots = np.vstack((np.cos(1.3 * t), np.sin(1.3 * t)))
model = DMD(rank=2, dt=0.1, backend="numpy").fit(snapshots)

reconstruction = model.reconstruct()  # includes training index zero
future = model.forecast(20)          # starts after the final training snapshot
print(model.eigenvalues_)
print(model.diagnostics())
```

For PyTorch CPU or CUDA, use the same estimator and array orientation:

```python
import torch
from modaldyn import DMD

x = torch.as_tensor(snapshots, dtype=torch.float64, device="cuda")
model = DMD(rank=2, backend="torch", device="cuda", dt=0.1).fit(x)
future = model.forecast(20)  # torch tensor on the selected device
```

An unavailable CUDA device raises an error. `backend="auto"` follows a tensor's
backend/device; NumPy input defaults to NumPy unless CUDA is explicitly requested.
Complex output is expected for oscillatory modes. Convert to real only when its
imaginary residual is negligible for your application.

## Select a variant

| Estimator | Intended use | Main assumption or cost |
| --- | --- | --- |
| `DMD` | Exact spectral DMD | Uniformly sampled snapshots; low-rank linear approximation |
| `TLSqDMD` | Total-least-squares preprocessing | Errors in both snapshot matrices; choose `tlsq_rank` |
| `FbDMD` | Forward/backward noise correction | Experimental principal-root convention; spectral ambiguity |
| `RandomizedDMD` | A seeded randomized range finder | Explicit integer rank; approximation trades accuracy for work |
| `HankelDMD` | Delay coordinates for partially observed dynamics | More memory; delay count selected on training/validation data |
| `DMDc` | Measured exogenous controls | Sufficiently exciting state/control data; dense state operator |
| `RobustDMD` | RPCA preprocessing followed by DMD | Low-rank signal plus sparse corruption; iterative dense SVDs |
| `StreamingDMD` | Incremental paired snapshots without history | Dense sufficient statistics use O(features²) memory |

These are established algorithms, not a claim to cover every published DMD
variant. Optimized/bagged DMD, EDMD, multiresolution and sparsity-promoting DMD
are outside this release. See [API and alignment rules](docs/api.md).

## What this package adds

GPU DMD already exists in flowTorch, and PyDMD has a substantially broader
variant catalogue. This project's value is its combination of a small NumPy-only
CPU installation, optional explicit device execution, shared input/rank checks,
inspectable diagnostics and a reproducible comparison harness. Diagnostics are
not unique to this package. No claim of universal speed or accuracy superiority
is made. The [source-backed comparison](docs/comparison.md) explains the evidence
and qualifications for PyDMD, PyKoopman, datafold and flowTorch.

Use `diagnostics()` to inspect the retained rank, backend/device and each
estimator's available numerical measures. Core DMD reports fitted one-step
relative error, retained conditioning and spectral radius. A small training
residual does not establish predictive performance; assess an untouched future
interval as well.

## Reproduce and contribute

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python examples/quickstart.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/benchmark.py \
  --backends numpy torch-cpu torch-cuda --features 256 --snapshots 160 \
  --rank 12 --repeats 5 --output benchmarks/results/local.json
```

Optional backends are explicitly skipped when unavailable. The benchmark records
software/hardware metadata, warmups, repetitions and accuracy; CUDA timings are
synchronized. Add `--pydmd` after installing PyDMD for a matched exact-DMD
comparison. See [benchmark methodology](docs/benchmarking.md).

CPU CI runs supported Python versions and a separate PyTorch CPU job. A manual
CUDA job needs a provisioned GPU runner; it fails if CUDA is unavailable.
Publications use an independently configured PyPI Trusted Publisher.

## Attribution

Original implementation by Giovanny Torres, released under the MIT license.
Algorithm and comparison references are listed in [docs/comparison.md](docs/comparison.md).
Please cite the relevant algorithm papers when using these methods in research.
