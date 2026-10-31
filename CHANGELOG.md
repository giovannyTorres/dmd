# Changelog

## 0.1.2 (2026-10-30)

- Promote mixed-precision snapshot pairs before estimating numerical rank.
- Require primal feasibility and dual stability in RobustDMD preprocessing;
  pause penalty growth while the components are still changing.
- Expose `rpca_error_` and `rpca_dual_error_` in fitted state and diagnostics.
- Scale norm calculations to avoid underflow/overflow in diagnostics.
- Reject invalid sampling/cutoff parameters consistently and reuse the
  backward operator's singular spectrum.
- Validate wheel and source installs separately, with no PyTorch dependency,
  runnable examples, version checks, dependency checks, and SHA-256 hashes.

## 0.1.1

Documentation-focused release. Algorithms and public APIs are unchanged.

- Clearer English package overview, installation and workflow guidance.
- Two reproducible examples with explicit input/output shapes and synthetic-data caveats.
- Practical application ideas, solver selection and CPU/CUDA validation limits.
- Absolute documentation links for PyPI and regression tests for README examples.

## 0.1.0

Initial alpha implementation, with original code and synthetic numerical validation.

- NumPy CPU and optional PyTorch CPU/CUDA backends, with explicit device failures.
- Exact, total least squares, forward/backward, randomized, Hankel, controlled,
  RPCA-preprocessed and sufficient-statistics streaming DMD.
- Numerical rank checks, JSON diagnostics and reproducible benchmark CLI.
- CPU parity tests, optional real-device CUDA tests, build and release workflows.

CUDA implementation is provided; this release was developed on a CPU-only host.
No CUDA timing or real-world accuracy advantage has been established.
