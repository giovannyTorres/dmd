# Reproduce measurements

The benchmark measures exact DMD on a synthetic low-rank oscillatory system.
Training and future holdout snapshots are contiguous in time. All implementations
receive the same data, requested rank, dtype and seed. The PyDMD baseline uses
`DMD(svd_rank=rank, exact=True, opt=False)`.
For held-out forecasting, all implementations project the final observed
training state onto fitted modes and evolve from that state. The PyDMD baseline
uses its public `modes` and `eigs` arrays for this calculation, so it has the same
initial condition as `modaldyn.forecast()`.

```bash
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/benchmark.py \
  --features 256 --snapshots 160 --rank 12 --repeats 5 \
  --output benchmarks/results/local-cpu.json
```

For the optional comparison install PyDMD separately. For PyTorch install the
appropriate CPU or CUDA build following <https://pytorch.org/get-started/locally/>.

```bash
python -m pip install pydmd
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/benchmark.py \
  --backends numpy torch-cpu torch-cuda --pydmd \
  --features 1024 --snapshots 512 --rank 24 --repeats 5 \
  --output benchmarks/results/local-comparison.json
```

The JSON records every repetition, median and minimum fit time, one-step,
reconstruction and held-out relative errors, package versions, CPU model, and
selected thread settings. CUDA runs synchronize before and after every timed
fit. Input conversion and explicit transfer to the target device occur before
timing; internal transfers performed by an implementation remain timed. Report
transfer-inclusive end-to-end measurements separately for a production workload.
Warnings are retained in each result row. PyDMD can warn about the condition
number of the exactly low-rank input; rank deficiency is intentional in this
synthetic fixture, and each implementation uses the specified truncated rank.

Unavailable optional backends are marked `skipped`, with a reason. Algorithm
failures are marked `error`, and the process returns a nonzero exit code. A
skipped CUDA row is not evidence of GPU correctness or performance. Peak CUDA
memory is PyTorch allocated memory including the resident input, not the whole
device's memory usage.

Interpret timings alongside accuracy. Implementations may perform different
validation, diagnostics, eigensolver work or allocations. The benchmark is a
small synthetic workload and cannot establish superiority for noisy physical
systems, very large datasets or another machine. Adjust size, rank, dtype,
thread counts and forecast horizon for your workload. GPU launch and transfer
costs can exceed savings on small matrices. No universal speedup is asserted.

For an application, split by time before tuning rank, delay count or other
parameters; tune with training/validation data and reserve an untouched test
interval. Do not select hyperparameters by repeatedly inspecting the test error.

## Recorded local run

The checked-in [`local-comparison.json`](../benchmarks/results/local-comparison.json)
was generated on 2026-10-29 using modaldyn 0.1.0, NumPy 2.3.5,
PyTorch 2.15.0+cpu and PyDMD 2025.8.1 on an AMD EPYC 9V74 CPU.
The signal has 256 features, 160 training snapshots, rank 12 and 20 future
holdout snapshots in float64. There is one warmup and five measured fits;
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

| Implementation | Median fit (ms) | Holdout relative error |
| --- | ---: | ---: |
| modaldyn NumPy CPU | 4.825 | 1.28e-14 |
| modaldyn PyTorch CPU | 6.231 | 1.34e-14 |
| PyDMD CPU | 6.001 | 1.09e-14 |
| modaldyn CUDA | skipped: unavailable | — |

These measurements describe this single synthetic fixture and shared machine.
The fit workloads include different internal checks, and five repetitions do
not establish a general performance advantage. This run supplies CPU numerical
evidence and a reproducible procedure; it supplies no CUDA measurement.
