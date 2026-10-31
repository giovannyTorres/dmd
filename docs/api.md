# API and array conventions

All estimators use column snapshots, shape `(n_features, n_snapshots)`. Controls
are `(n_controls, n_pairs)`. Sampling is uniform; `dt` changes physical frequency
and growth-rate units, not the discrete operator. Input arrays must be finite.
Integer inputs become float64; half precision is promoted to float32. Eigenmodes
may be complex even when all measurements are real. Mode order, phase and scaling
need not match across eigensolvers; compare spectra up to permutation and compare
reconstructed dynamics rather than raw mode columns.

Explicit X/Y pairs are promoted to a common dtype before computing numerical
rank on both NumPy and PyTorch. In particular, float64 targets promote float32
inputs to float64; information already rounded out of the inputs is not restored.

`backend="numpy"` selects CPU, `backend="torch"` selects PyTorch, and
`backend="auto"` follows a torch input or an explicitly requested CUDA device;
otherwise it selects NumPy. A torch input keeps its device when none is given.
Explicit CUDA requests fail if unsupported. AMD ROCm, MPS, distributed/multi-GPU
and automatic differentiation are not supported targets of this release.

## DMD family

```python
DMD(rank=None, *, backend="auto", device=None, dt=1.0,
    tlsq_rank=0, forward_backward=False, randomized=False,
    oversampling=10, power_iters=2, seed=0, rcond=None)
TLSqDMD(rank=None, *, tlsq_rank=2, **dmd_options)
FbDMD(rank=None, **dmd_options)
RandomizedDMD(rank, **dmd_options)
```

`rank=None` retains numerical rank; a positive integer caps it; a float in
`(0, 1]` specifies retained squared-singular-value energy. `1` is an integer
rank of one, while `1.0` means all numerical energy. Requested integer ranks are
capped at the estimated numerical rank. `rcond` overrides the relative singular
value cutoff. Randomized DMD requires an integer rank and uses a local seeded
random range finder; it does not alter the global random seed.
For randomized fitting, `numerical_rank_` is measured in the sketch, not the
complete input matrix. Diagnostics label this scope and warn that omitted
directions may remain undetected.

| Method / attribute | Meaning |
| --- | --- |
| `fit(X)` | Consecutive snapshots; forms `X[:, :-1]` and `X[:, 1:]` |
| `fit(X, Y)` | Explicit one-step pairs, identical shapes; may be unordered |
| `predict(X)` | One spectral-operator step for a vector or matrix |
| `reconstruct()` | Modal rollout from the first training state, training length |
| `reconstruct(n)` | First `n` snapshots starting at index zero |
| `reconstruct(indices)` | Nonnegative integer step indices from the same initial state |
| `forecast(n)` | Next `n` states after the final training target; current state excluded |
| `eigenvalues_`, `modes_`, `amplitudes_` | Fitted discrete spectrum, modes and first-state amplitudes |
| `frequencies_` | `angle(eigenvalue)/(2*pi*dt)`; principal phase, cycles per time unit |
| `growth_rates_` | `log(abs(eigenvalue))/dt`; a zero eigenvalue has negative infinite growth |
| `diagnostics()` | JSON-compatible scalar metadata and numerical measures |

For arbitrary explicit pairs, modal rollout still starts at the first X column;
it does not reconstruct the pair ordering. The default length is number of
pairs plus one. A zero-length forecast has shape `(n_features, 0)`.

Core one-step relative error is `norm(predict(X) - Y) / norm(Y)` on the original
training pairs; it is `None` if the target norm is zero. Retained condition
number uses the fitted singular spectrum. Mode conditioning and the initial
state representation error help identify a poorly resolved modal basis.
An infinite modal condition number is returned as `None` with a diagnostic note.
The spectral operator is `Phi @ diag(lambda) @ pinv(Phi)`; there is no
direct-regression fallback for defective/Jordan operators. These diagnostics may synchronize
CUDA and return scalar metadata to the host. Arrays remain on the selected
backend; a fitted model may retain training matrices for diagnostics.

Forward/backward DMD uses a principal matrix square root. Negative-real spectra
and root-branch choices can produce an ambiguous result; singular backward
operators are rejected. Treat this option as experimental.

## Delay embedding

```python
delay_embedding(X, delays=2, *, backend="auto", device=None)
HankelDMD(delays=2, rank=None, **dmd_options)
```

For `(n, m)` input, `delay_embedding` allocates `(n*delays, m-delays+1)`, with
the oldest sample in the first block. `HankelDMD.fit(X)` requires at least
`delays+1` snapshots. `reconstruct(steps=None)` averages overlapping physical
snapshots. Here `steps` accepts only a contiguous integer length at least
`delays`; arbitrary index arrays are not supported. `predict(window)` requires
`(n_features, delays)` and returns the next physical state. `forecast(n)` returns
future physical states after training. `modes_` is the first physical block;
`embedded_modes_` is the complete delay-space mode matrix.

## Measured controls

```python
DMDc(rank=None, *, backend="auto", device=None, dt=1.0)
model.fit(X, U, Y=None, *, B=None)
```

The model is `x[k+1] = A_ @ x[k] + B_ @ u[k]`. With consecutive X, controls
contain `X.shape[1]-1` columns. With explicit pairs, X, Y and U have the same
column count. `B` supplies a known control operator; otherwise state and control
operators are jointly regressed. `rank` truncates the regression input; this is
a dense-state implementation, not reduced-order DMDc. Excitation and sample
coverage must support identification.

- `predict(X, U)` applies one step for vectors or matching matrices.
- `simulate(x0, U)` includes `x0`, then one column per control.
- `forecast(U)` starts after the last training target and excludes that target.
- `reconstruct()` simulates the training controls and requires consecutive X.
- `diagnostics()` reports regression rank, dimensions, relative residual and
  whether the data and retained rank support full operator identification.
  `full_regression_rank` is the input's numerical rank;
  `required_regression_rank` is features+controls (features if B is known).
  A small residual alone does not establish identifiability.

## Low rank plus sparse corruption

```python
RobustDMD(rank=None, *, rpca_lambda=None, rpca_tol=1e-7,
          rpca_max_iter=1000, **dmd_options)
```

`fit(X)` first performs principal component pursuit using inexact augmented
Lagrange iterations, then fits DMD to `low_rank_`. It accepts consecutive X
only. The default sparsity penalty is `1/sqrt(max(X.shape))`. Inspect
`sparse_`, `rpca_converged_`, `rpca_iterations_` and `rpca_residual_`; reaching
the iteration limit emits a warning. This two-stage pipeline assumes low-rank
structure and sparse corruption. Its reconstruction/forecast describe the
estimated clean component. Its DMD residual is computed on that component;
the RPCA residual is a separate feasibility measure. `rpca_dual_error_` measures
normalized component change, and `rpca_error_` is the maximum of the primal
residual and dual error. Both must be at most `rpca_tol` for convergence.
The penalty increases only while primal error exceeds dual error; a small
feasibility residual alone is insufficient. These values also appear in
`diagnostics()`. It is not guaranteed to
improve dense-noise estimation or held-out forecasts.

## Streaming sufficient statistics

```python
StreamingDMD(rank=None, *, backend="auto", device=None, dt=1.0,
             forgetting_factor=1.0, rcond=None)
```

`partial_fit(X, Y=None)` accumulates `G = sum(x x*)` and `H = sum(y x*)`.
When Y is omitted, only transitions inside that batch are used. Batch boundaries
are **not joined**: pass explicit transition pairs to include a boundary.
`forgetting_factor` discounts previous statistics once per accepted batch,
regardless of batch length, so changing batching changes the weighting.

Call `finalize()` after the last batch to compute `operator_ = H @ pinv(G)`.
Each new partial fit invalidates finalized outputs. `fit(X, Y=None)` resets and
finalizes in one call. `rank` is None or a positive integer and truncates G's
pseudoinverse; it does not reduce O(n_features�) statistic storage. Forming G
squares the condition number of the snapshots, so batch SVD can be preferable.

`predict(X)` advances states one step. `forecast(x0, n_steps)` explicitly takes
the starting state, excludes it, and returns future columns. This model keeps
no snapshot history and has no training reconstruction method. Its `modes_`
contains the full operator's eigenvectors, including null directions, and may
have more columns than `rank_`. Forecasting uses the operator directly.
