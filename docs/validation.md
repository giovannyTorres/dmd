# Validation records

## 0.1.2 local validation

Local Windows checks on 2026-10-30:

| Environment | Result |
| --- | --- |
| Python 3.13.5, NumPy 2.5.4, PyTorch 2.15.0+cpu | 102 passed, 8 CUDA tests skipped |
| Python 3.10.18, NumPy 1.24.0, no PyTorch | 74 passed, 36 optional tests skipped |

The full CPU suite passes with warnings treated as errors. Source coverage is
95.6% (674/705 statements); Ruff also passes. The new regressions cover mixed
precision rank, primal/dual RPCA convergence with real and complex inputs,
diagnostic scaling, and invalid numerical parameters.

For the seeded sparse-corruption case in `tests/test_regressions.py`, the
original implementation declared convergence with relative clean-component
error about `2.24e-3`. The corrected version reaches about `3.50e-9` on this
host. The stricter criterion uses more iterations in this case. This is a
synthetic regression, not a universal recovery or performance guarantee.

Run `python scripts/check_artifacts.py --output artifact-validation.json` after
building to compare shipped sources and validate wheel/source installs in
separate base-only environments. Hashes identify the exact artifacts tested.
This local record predates remote CI. Real CUDA execution was not tested.

## Historical 0.1.0 record

Local verification on 2026-10-29, Python 3.12.14, NumPy 2.3.5,
PyTorch 2.15.0+cpu. Commands were run against the release source:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -rs
python -m ruff check .
python examples/quickstart.py
python examples/controlled.py
python -m build
python -m twine check dist/*
```

- 87 tests passed; 8 CUDA tests skipped because no CUDA device was available.
- Ruff passed. Both wheel and source distribution passed `twine check`.
- Exact DMD 20-step synthetic holdout relative error: 2.65e-15.
- DMDc 40-step synthetic holdout relative error: 2.40e-15.
- A clean virtual environment installed the wheel with NumPy 2.5.4 and no
  PyTorch, imported all public estimators and ran both examples successfully.
- Tests cover known dynamics, complex and single-precision inputs, backend
  parity, rank and state validation, sparse RPCA corruption, control
  identifiability, streaming batches and documented modal limitations.

The complete exact-DMD comparison with PyDMD and raw timings is in
[`benchmarks/results/local-comparison.json`](../benchmarks/results/local-comparison.json).
These are synthetic correctness checks, not evidence of real-world impact,
universal performance or GPU validation. Remote CI is tracked separately in
[GitHub Actions](https://github.com/giovannyTorres/dmd/actions).
