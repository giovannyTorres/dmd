# DMD library landscape and a defensible value proposition

Evidence checked on **2026-10-29** against public project READMEs, source exports and PyPI JSON metadata. This is a focused comparison, not an exhaustive survey or performance ranking. The algorithms below are established research. CPU/GPU DMD is already available elsewhere; this project must not claim to invent it.

## What is already published

| Project | Observed PyPI release | Documented strengths | Backend evidence and qualification |
| --- | --- | --- | --- |
| [PyDMD](https://pydmd.github.io/PyDMD/) | 2025.8.1, Python >=3.8 | Broad DMD catalogue: exact/projected, forward-backward, TLS preprocessing, compressed/randomized, Hankel/higher-order, multiresolution, control, sparsity-promoting, physics-informed, optimized and bagged optimized DMD; tutorials and plotting. | README describes NumPy/SciPy core. Its exported modules and README do not establish a uniform CUDA backend for every variant. This is not proof that all GPU-related integrations are absent. |
| [PyKoopman](https://pykoopman.readthedocs.io/) | 1.2.1, Python >=3.11 | Koopman framework with observable dictionaries, polynomial/RBF/time-delay/random-Fourier features; DMD/PyDMD wrapper, DMDc, EDMD/EDMDc, KDMD, HAVOK and neural-network DMD. | README explicitly describes installation with GPU support via CUDA-enabled PyTorch. The presence of PyTorch does not establish GPU execution for its NumPy/PyDMD regressors; support depends on the estimator. |
| [datafold](https://datafold-dev.gitlab.io/datafold/) | 2.0.2, Python >=3.9 | DMD, online/streaming DMD, DMDControl, EDMD, time-series collections, temporal cross-validation, manifold methods and control workflows. | README describes NumPy/SciPy/pandas/scikit-learn integration. A uniform CUDA backend is not established by the inspected documentation. The upstream README itself qualifies interface stability and production readiness. |
| [flowTorch](https://flowtorch.readthedocs.io/) | `flowtorch-fluid` 1.10.0, Python >=3.10 | Fluid-data loading and reduced-order modeling; DMD with TLS, optimal amplitudes and unitary constraints; HODMD, OptDMD and HOOptDMD exports; extensive SVD/SPOD and field tooling. | README explicitly states PyTorch CPU/GPU and single/double precision. Its DMD source operates on PyTorch tensors. The optional forward-backward-consistent ADMM helper calls SciPy and `.numpy()`, so this particular implementation path should not be generalized as CUDA-native without testing. |

Release versions and dependency metadata are from PyPI; feature evidence is from the inspected source branch. **A branch feature is not automatically part of the named PyPI release.** The table does not claim that every feature was runtime-tested. Comparisons should be rechecked when publishing subsequent releases.

## Where a new library can provide value

1. **A small NumPy-only CPU installation with an optional, explicit PyTorch CPU/CUDA backend across the selected variants.** This combines deployment flexibility and a predictable device API. It differentiates the package's scope and packaging; GPU support itself is not novel. The CUDA path should reject unavailable devices clearly and avoid silently converting full input matrices to NumPy.
2. **Inspectable numerical diagnostics as part of the normal fit.** Report retained rank, singular spectrum/conditioning, fitted one-step residual and reconstruction error with exact definitions. Include finite-input checks, effective numerical rank truncation and explicit zero-eigenvalue behavior. Competitors also expose diagnostics; the value is a consistent, concise interface across this project's variants, not exclusive ownership of diagnostics.
3. **A reproducible comparison harness.** Record shape, rank, dtype, seeds, backend, device, software versions, warm-up count, repeats, CUDA synchronization and whether host/device transfer is timed. Show accuracy beside time. Compare matched exact/projected settings against PyDMD. This is a verification tool, not a speed claim. Transfer and launch overhead can make GPU slower for small problems.
4. **Portable fitted artifacts and runnable examples.** A documented, non-pickle serialization format can move fitted spectral models between CPU and GPU. Explain the schema and limits; do not imply that loading a spectrum permits resuming every training procedure. Examples should cover an oscillator, a controlled system, noisy data and out-of-sample forecasting.
5. **Honest variant selection and boundaries.** Exact/projected for basic dynamics; TLS for errors in both snapshot matrices; Hankel delay coordinates for underobserved oscillations; randomized compression for tall low-rank data; DMDc for measured exogenous controls. These are practical recommendations with assumptions, not guarantees of superiority.

A useful positioning statement is: **“A compact DMD toolkit with a NumPy CPU core, optional PyTorch CPU/CUDA execution, consistent numerical diagnostics and reproducible validation.”** The final README must list only implemented and tested capabilities. Do not promise all variants in the literature.

## Claims to avoid

- “The first DMD library for GPU,” “faster than PyDMD” without matched measured evidence, or “more accurate” based only on training reconstruction.
- “Robust DMD” as a synonym for TLS: TLS addresses a particular noise model and is not RPCA/outlier rejection.
- “Randomized DMD is always faster”: a full SVD followed by truncation is not randomized acceleration, and randomized methods can lose accuracy.
- “Supports all GPUs”: a CUDA implementation targets supported NVIDIA hardware and installed PyTorch/CUDA combinations. AMD ROCm, Apple MPS, multi-GPU and automatic differentiation require their own verified support.
- “CPU/GPU verified” when CUDA was skipped because no device was available. Report that distinction prominently.
- “Production ready,” “stable” or a research novelty claim solely because unit tests pass.

## Validation that would make the differentiation credible

- Recover a known linear operator's spectrum up to permutation; compare phase-invariant reconstructions rather than raw complex mode columns.
- Test real and complex data, float32/float64, finite-input rejection, rank-deficient matrices, zero eigenvalues and clear shape/orientation errors.
- Evaluate held-out forecast error separately from in-sample reconstruction. For DMDc, use persistently exciting controls; an autonomous trajectory may not identify the control matrix.
- Test each supported variant on NumPy and PyTorch CPU; run the same suite on CUDA when hardware is available. Keep skipped CUDA results visible.
- Compare noise methods across multiple seeded trials and a specified sensor-noise model. Never claim a noise benefit from one favorable synthetic seed.
- Build wheel and source distribution, install the wheel in a fresh environment, run examples from outside the checkout, check metadata, and make release provenance inspectable.

## Name check

Both [modaldyn](https://pypi.org/pypi/modaldyn/json) and [dmdforge](https://pypi.org/pypi/dmdforge/json) returned **HTTP 404** during this check. This indicates no currently visible PyPI project at those exact JSON endpoints; it is **not** a reservation, trademark clearance or guarantee that upload will be accepted. GitHub repository-name availability depends on the selected account.

## Sources and inspection anchors

- PyDMD [README](https://github.com/PyDMD/PyDMD/blob/024464ff667e4b0226f9ecbe933bc67f3fe21e81/README.md), [module exports](https://github.com/PyDMD/PyDMD/blob/024464ff667e4b0226f9ecbe933bc67f3fe21e81/pydmd/__init__.py), [PyPI metadata](https://pypi.org/pypi/pydmd/json).
- PyKoopman [README](https://github.com/dynamicslab/pykoopman/blob/81c640f2d92c01ebd4f6087f76a050ea622b3216/README.md), [regressor exports](https://github.com/dynamicslab/pykoopman/blob/81c640f2d92c01ebd4f6087f76a050ea622b3216/src/pykoopman/regression/__init__.py), [PyPI metadata](https://pypi.org/pypi/pykoopman/json).
- datafold [README mirror](https://github.com/datafold-dev/datafold/blob/e4907dc919579284751277fb9afafc653f71d85a/README.rst), [canonical repository](https://gitlab.com/datafold-dev/datafold), [PyPI metadata](https://pypi.org/pypi/datafold/json).
- flowTorch [README](https://github.com/AndreWeiner/flowtorch/blob/f2e515c485647c76c1ff3c7340b2900ba93e4724/README.md), [DMD source](https://github.com/AndreWeiner/flowtorch/blob/f2e515c485647c76c1ff3c7340b2900ba93e4724/flowtorch/analysis/dmd.py), [analysis exports](https://github.com/AndreWeiner/flowtorch/blob/f2e515c485647c76c1ff3c7340b2900ba93e4724/flowtorch/analysis/__init__.py), [PyPI metadata](https://pypi.org/pypi/flowtorch-fluid/json). The legacy FlowModelingControl URL redirects to AndreWeiner.

## Algorithm references

- Schmid (2010), *Dynamic mode decomposition of numerical and experimental data*. [doi:10.1017/S0022112010001217](https://doi.org/10.1017/S0022112010001217).
- Tu et al. (2014), *On Dynamic Mode Decomposition: Theory and Applications*. [doi:10.3934/jcd.2014.1.391](https://doi.org/10.3934/jcd.2014.1.391).
- Hemati et al. (2017), *De-biasing the dynamic mode decomposition for applied Koopman spectral analysis of noisy datasets*. [doi:10.1007/s00162-017-0432-2](https://doi.org/10.1007/s00162-017-0432-2).
- Dawson et al. (2016), *Characterizing and correcting for the effect of sensor noise in the dynamic mode decomposition*. [doi:10.1007/s00348-016-2127-7](https://doi.org/10.1007/s00348-016-2127-7).
- Erichson et al. (2019), *Randomized dynamic mode decomposition*. [doi:10.1137/18M1215013](https://doi.org/10.1137/18M1215013).
- Le Clainche and Vega (2017), *Higher order dynamic mode decomposition*. [doi:10.1137/15M1054924](https://doi.org/10.1137/15M1054924). Delay/Hankel DMD and full HODMD must not be treated as automatically identical implementations.
- Proctor et al. (2016), *Dynamic mode decomposition with control*. [doi:10.1137/15M1013857](https://doi.org/10.1137/15M1013857).
- Hirsh et al. (2020), *Centering data improves the dynamic mode decomposition*. [doi:10.1137/19M1289881](https://doi.org/10.1137/19M1289881).
- Demo et al. (2018), PyDMD software paper. [doi:10.21105/joss.00530](https://doi.org/10.21105/joss.00530).
- Ichinaga et al. (2024), *PyDMD: A Python Package for Robust Dynamic Mode Decomposition*. [JMLR 25, paper 24-0739](https://jmlr.org/papers/v25/24-0739.html).
- Pan et al. (2024), PyKoopman software paper. [doi:10.21105/joss.05881](https://doi.org/10.21105/joss.05881).
- Lehmberg et al. (2020), datafold software paper. [doi:10.21105/joss.02283](https://doi.org/10.21105/joss.02283).
