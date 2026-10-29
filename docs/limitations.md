# Scope and limitations

Version 0.1 is a research implementation. A distributable wheel, passing tests
and a GitHub release do not establish suitability for a particular experiment
or deployment. Evaluate prediction on held-out time intervals and compare with
appropriate baselines before relying on the identified dynamics.

## Numerical assumptions

- Snapshots must be uniformly sampled, correctly oriented, finite and aligned.
  No irregular-time optimized DMD is implemented.
- Rank selection is consequential. Retained energy is descriptive, not a
  forecasting guarantee, and truncated singular directions cannot be recovered.
- Eigendecompositions are sensitive for nearly repeated or defective operators.
  A modal representation may be ill-conditioned even with a small regression
  residual. Core `predict` and modal rollouts use
  `Phi @ diag(lambda) @ pinv(Phi)`; no direct-regression fallback is provided.
  Defective/Jordan-chain dynamics therefore cannot in general be represented.
  Inspect modal conditioning and the initial-state representation error.
- Frequencies use the principal complex phase and are subject to sampling
  aliasing. No physical frequency is identifiable beyond the sampling limit.
- Unstable eigenvalues produce growing forecasts. The package reports this in
  core diagnostics; it does not stabilize spectra automatically.
- Forward/backward DMD uses the principal matrix square root. Root selection
  loses sign/branch information for some spectra; singular inverse dynamics
  are unsupported. This variant is experimental.
- TLS and RPCA model different error structures. RPCA assumes low-rank signal
  with sparse corruption and can fail or remove useful dynamics when those
  assumptions do not hold. Neither approach guarantees more accurate forecasts.

## Devices and memory

NumPy supports CPU. PyTorch supports the selected CPU/CUDA execution paths and
requires a compatible installed build and driver. CUDA support is implemented;
skipped CUDA tests do not verify it on hardware. Run the manual GPU suite and
benchmark on your intended machine. No claim is made for ROCm, MPS, TPU,
multi-GPU, distributed execution, autograd or end-to-end differentiability.

Full matrices are not silently sent to NumPy for CUDA linear algebra. Scalar
validation and diagnostics synchronize the device; energy-rank selection may
copy a singular-value vector to the host. Randomized sketches are seeded on CPU
and transferred to the selected backend. GPU throughput must include these
costs when assessing real workloads.

This release uses dense algebra. DMD may retain training pairs for diagnostics;
Hankel allocates delay embeddings; RPCA performs repeated SVDs; DMDc stores dense
A and B. Streaming avoids snapshot history but retains two dense square state
statistics and eventually a dense operator. It is not an out-of-core,
sparse-state or fixed-low-rank memory algorithm. Gram statistics also square
the snapshot condition number.

## Feature and publication boundaries

Optimized/bagged DMD, EDMD, sparse-promoting DMD, multiresolution, full HODMD,
physics constraints, distributed decomposition and a scikit-learn estimator
protocol are not claimed for this release. Hankel delay DMD is not synonymous
with every algorithm published as HODMD. RPCA followed by DMD is an established
pipeline, not a new estimator claimed by this project.

Benchmarks use synthetic data and the stated hardware/settings. They do not
establish universal speedups or application impact. Package publication on
PyPI requires separate authentication and publisher setup; the repository's
release workflow alone does not mean a PyPI release exists.
