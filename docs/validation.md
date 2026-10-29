# Validation record for 0.1.0

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
