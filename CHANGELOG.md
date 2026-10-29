# Changelog

## 0.1.0

Initial alpha implementation, with original code and synthetic numerical validation.

- NumPy CPU and optional PyTorch CPU/CUDA backends, with explicit device failures.
- Exact, total least squares, forward/backward, randomized, Hankel, controlled,
  RPCA-preprocessed and sufficient-statistics streaming DMD.
- Numerical rank checks, JSON diagnostics and reproducible benchmark CLI.
- CPU parity tests, optional real-device CUDA tests, build and release workflows.

CUDA implementation is provided; this release was developed on a CPU-only host.
No CUDA timing or real-world accuracy advantage has been established.
