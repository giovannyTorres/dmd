# ModalDyn

**Turn time-ordered snapshots into interpretable modes and short-horizon forecasts.**

ModalDyn is a compact Dynamic Mode Decomposition (DMD) toolkit for researchers and
engineers who want to understand evolving systems, not just fit a black box.
Start with NumPy, inspect the model's diagnostics, and use the same estimator API
with optional PyTorch CPU or CUDA execution.

[PyPI](https://pypi.org/project/modaldyn/) ·
[API guide](https://github.com/giovannyTorres/dmd/blob/main/docs/api.md) ·
[Examples](https://github.com/giovannyTorres/dmd/tree/main/examples) ·
[Limitations](https://github.com/giovannyTorres/dmd/blob/main/docs/limitations.md)

## Why ModalDyn?

- **A small starting point.** The basic installation needs only NumPy; PyTorch is optional.
- **A model you can inspect.** Examine eigenvalues, modes, numerical rank,
  conditioning and fitted one-step error instead of relying on a forecast alone.
- **Several workflows, one package.** Reconstruct snapshots, forecast dynamics,
  include measured controls, or add delay coordinates for partial observations.
- **Explicit computation.** Choose the backend and device. A request for unavailable
  CUDA fails clearly rather than silently falling back to CPU.

### Where it fits

These are application ideas, not claims of validation on production datasets:

| Your data | A useful starting question |
| --- | --- |
| Time-ordered fluid or simulation fields | Which coherent patterns evolve together, and at which rates? |
| Vibration or multi-sensor recordings | Can a low-dimensional model describe oscillations and forecast a held-out interval? |
| Measurements from an actuated experiment | How do the observed states respond to known control inputs? |

DMD approximates dynamics with a linear evolution model. It is especially useful
when a few modes explain the behavior over the time window you care about.
Choose rank and other settings on training/validation data, then test a future
interval that the model has never seen.

## Install

Python **3.10+** and NumPy **1.24+** are required.

```bash
python -m pip install modaldyn
```

Optional PyTorch support:

```bash
python -m pip install 'modaldyn[torch]'
```

For a specific CPU or CUDA build, follow the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/).
The extra does not install a GPU driver or guarantee CUDA availability.

## Quickstart: forecast a held-out oscillation

Each **column is a snapshot** and each **row is a feature**. In this reproducible
synthetic example, two sensor channels follow a damped oscillation. The first
80 snapshots train the model; the last 20 are used only for evaluation.

```python
import numpy as np
from modaldyn import DMD

angle = 0.16
operator = 0.995 * np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)],
])
snapshots = np.empty((2, 100))  # 2 channels, 100 equally spaced observations
snapshots[:, 0] = [1.0, 0.2]
for k in range(99):
    snapshots[:, k + 1] = operator @ snapshots[:, k]

model = DMD(rank=2, dt=0.1).fit(snapshots[:, :80])
reconstruction = model.reconstruct()  # shape (2, 80), includes the first snapshot
forecast = model.forecast(20)        # shape (2, 20), starts AFTER the last training point
truth = snapshots[:, 80:]
relative_error = np.linalg.norm(forecast - truth) / np.linalg.norm(truth)

print("Reconstruction:", reconstruction.shape)
print("Forecast:", forecast.shape)
print("Held-out relative error:", f"{relative_error:.2e}")
print("Retained rank:", model.diagnostics()["rank"])
assert relative_error < 1e-10  # check for this noiseless linear example only
```

The output shapes are `(2, 80)` and `(2, 20)`, with retained rank `2`.
The error is near numerical precision because the example follows an exact
linear model. This is a reproducibility check, **not a real-world accuracy claim**.
For measured data, compare against a simple baseline and evaluate multiple
held-out windows. Complex-valued modes and forecasts are normal for oscillations;
only discard the imaginary part when its residual is negligible.

## Example: forecast a system with known controls

Suppose an experiment records two state variables and one actuator input.
`DMDc` fits the relationship `x[k+1] = A @ x[k] + B @ u[k]`.
This synthetic example shows the required alignment: **81 state snapshots pair
with 80 control inputs**, one input for each transition.

```python
import numpy as np
from modaldyn import DMDc

rng = np.random.default_rng(12)
A = np.array([[0.93, 0.08], [-0.05, 0.97]])
B = np.array([[0.15], [0.40]])
controls = rng.standard_normal((1, 120))
states = np.empty((2, 121))
states[:, 0] = [0.4, -0.2]
for k in range(120):
    states[:, k + 1] = A @ states[:, k] + B @ controls[:, k]

model = DMDc().fit(states[:, :81], controls[:, :80])
forecast = model.forecast(controls[:, 80:])  # 40 supplied FUTURE control inputs
truth = states[:, 81:]
relative_error = np.linalg.norm(forecast - truth) / np.linalg.norm(truth)

print("Controlled forecast:", forecast.shape)  # (2, 40)
print("Held-out relative error:", f"{relative_error:.2e}")
assert forecast.shape == truth.shape
assert relative_error < 1e-10
```

The future controls are inputs you provide, not values predicted by the model.
Reliable identification needs sufficiently informative state/control data.
This noiseless example verifies the API and alignment; it does not establish
performance on a physical experiment.

## Pick the right estimator

| Estimator | Use it when | Keep in mind |
| --- | --- | --- |
| `DMD` | You want an exact-DMD baseline | Uniform time spacing and a useful low-rank linear approximation |
| `TLSqDMD` / `FbDMD` | You are exploring noise-aware formulations | Tune TLS rank; forward/backward DMD has experimental spectral-root limits |
| `RandomizedDMD` | You want a seeded low-rank approximation | Specify an integer rank; measure the accuracy/cost trade-off |
| `HankelDMD` | Delay coordinates may expose hidden dynamics | More memory; choose delays without leaking test data |
| `DMDc` | Measured controls influence the evolution | Align each control with its state transition |
| `RobustDMD` | Sparse corruptions motivate RPCA preprocessing | Assumes low-rank structure plus sparse corruption; iterative dense SVDs |
| `StreamingDMD` | You cannot retain the full snapshot history | Dense sufficient statistics still need O(features²) memory |

See the [API guide](https://github.com/giovannyTorres/dmd/blob/main/docs/api.md)
for parameters, alignment rules and variant-specific diagnostics.

## CPU, PyTorch and CUDA

The examples above use NumPy on CPU. For the PyTorch backend, use
`backend="torch", device="cpu"` in the estimator; `device="cuda"` requests CUDA
explicitly. Forecasts use the selected backend. `backend="auto"` follows an
input tensor's backend/device; NumPy input defaults to NumPy unless CUDA is
explicitly requested.

**Validation:** CI exercises NumPy on Python 3.10–3.13 and PyTorch on CPU.
CUDA code and optional tests are included, but GPU hardware correctness and
performance have **not** been established for this release. Skipped CUDA tests
are not GPU validation. Small problems may be faster on CPU because transfers
and synchronization have a cost.

## Scope and trade-offs

This is an **alpha research package**, not a universal forecasting solution.
A small training residual does not guarantee useful future predictions.
Nonstationarity, noise, rank choice and insufficient measurements can all change
results. Read the [limitations](https://github.com/giovannyTorres/dmd/blob/main/docs/limitations.md)
before interpreting a model.

ModalDyn favors a compact NumPy-first API and explicit diagnostics/device choice.
PyDMD offers a broader variant catalogue; GPU-enabled DMD also exists elsewhere.
There is no claim of universal speed or accuracy superiority. The
[source-backed comparison](https://github.com/giovannyTorres/dmd/blob/main/docs/comparison.md)
covers PyDMD, PyKoopman, datafold and flowTorch. Optimized/bagged DMD, EDMD,
multiresolution DMD and sparsity-promoting DMD are outside this release.

## Learn more and contribute

- [API reference](https://github.com/giovannyTorres/dmd/blob/main/docs/api.md)
- [Runnable examples](https://github.com/giovannyTorres/dmd/tree/main/examples)
- [Validation notes](https://github.com/giovannyTorres/dmd/blob/main/docs/validation.md)
- [Reproducible benchmarking](https://github.com/giovannyTorres/dmd/blob/main/docs/benchmarking.md)
- [Report an issue](https://github.com/giovannyTorres/dmd/issues)
- [Release guide](https://github.com/giovannyTorres/dmd/blob/main/docs/releasing.md)

Original implementation by Giovanny Torres, released under the
[MIT license](https://github.com/giovannyTorres/dmd/blob/main/LICENSE).
Please cite the relevant algorithm papers listed in the
[comparison and references](https://github.com/giovannyTorres/dmd/blob/main/docs/comparison.md)
when using these methods in research.
