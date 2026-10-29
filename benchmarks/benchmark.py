#!/usr/bin/env python3
"""Reproducible exact-DMD fit benchmark; see docs/benchmarking.md.

Run after installing modaldyn: python benchmarks/benchmark.py --backends numpy
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modaldyn import DMD


def synthetic_data(features, snapshots, rank, seed, dtype):
    """Stable, real, exactly rank-r dynamics with conjugate oscillatory modes."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((features, rank)))
    operator = np.zeros((rank, rank))
    for pair, start in enumerate(range(0, rank - 1, 2)):
        theta = 0.08 + 1.2 * (pair + 1) / max(1, rank // 2)
        decay = 0.998 - 0.0003 * pair / max(1, rank // 2)
        c, s = np.cos(theta), np.sin(theta)
        operator[start:start + 2, start:start + 2] = decay * np.array([[c, -s], [s, c]])
    if rank % 2:
        operator[-1, -1] = 0.999
    latent = np.empty((rank, snapshots))
    latent[:, 0] = rng.standard_normal(rank)
    for index in range(1, snapshots):
        latent[:, index] = operator @ latent[:, index - 1]
    return (basis @ latent).astype(dtype)


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def relative_error(actual, expected):
    value = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    if not np.isfinite(value):
        raise FloatingPointError('Nonfinite relative error')
    return value


def to_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def benchmark_backend(name, training, holdout, args):
    torch = None
    cuda = name == 'torch-cuda'
    if name.startswith('torch'):
        try:
            import torch
        except ImportError:
            return {'backend': name, 'status': 'skipped', 'reason': 'PyTorch is not installed'}
        if cuda and not torch.cuda.is_available():
            return {'backend': name, 'status': 'skipped', 'reason': 'CUDA is unavailable'}
        device = 'cuda' if cuda else 'cpu'
        data = torch.as_tensor(training, device=device)

        def factory():
            return DMD(rank=args.rank, backend='torch', device=device)
    elif name == 'pydmd':
        try:
            from pydmd import DMD as PyDMD
        except ImportError:
            return {'backend': name, 'status': 'skipped', 'reason': 'PyDMD is not installed'}
        data = training

        def factory():
            return PyDMD(svd_rank=args.rank, exact=True, opt=False)
    else:
        data = training

        def factory():
            return DMD(rank=args.rank, backend='numpy')

    def synchronize():
        if cuda:
            torch.cuda.synchronize()

    captured_warnings = set()

    def fit(model):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            model.fit(data)
        captured_warnings.update(f'{record.category.__name__}: {record.message}' for record in records)

    for _ in range(args.warmups):
        model = factory()
        fit(model)
        synchronize()
        del model
    timings = []
    peaks = []
    for _ in range(args.repeats):
        model = factory()
        synchronize()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        fit(model)
        synchronize()
        timings.append(time.perf_counter() - start)
        if cuda:
            peaks.append(torch.cuda.max_memory_allocated())
        # Delete each preceding model before the next repetition. Only the last
        # fit remains for numerical checks, which are outside the timing window.
        if len(timings) < args.repeats:
            del model

    prediction = to_numpy(model.predict(data[:, :-1]))
    if name == 'pydmd':
        reconstructed = np.asarray(model.reconstructed_data)
        eigenvalues = np.asarray(model.eigs)
        # Use identical forecast anchoring across implementations. PyDMD's
        # dmd_time extrapolation instead evolves its first-snapshot amplitude.
        amplitudes = np.linalg.lstsq(model.modes, training[:, -1], rcond=None)[0]
        steps = np.arange(1, holdout.shape[1] + 1)
        forecast = model.modes @ (amplitudes[:, None] * eigenvalues[:, None] ** steps[None, :])
    else:
        reconstructed = to_numpy(model.reconstruct())
        forecast = to_numpy(model.forecast(holdout.shape[1]))
        eigenvalues = to_numpy(model.eigenvalues_)
    result = {
        'backend': name,
        'status': 'ok',
        'fit_seconds': timings,
        'median_fit_seconds': statistics.median(timings),
        'min_fit_seconds': min(timings),
        'one_step_relative_error': relative_error(prediction, training[:, 1:]),
        'reconstruction_relative_error': relative_error(reconstructed, training),
        'holdout_relative_error': relative_error(forecast, holdout),
        'eigenvalue_count': int(eigenvalues.size),
        'warnings': sorted(captured_warnings),
    }
    if peaks:
        result['peak_cuda_allocated_bytes'] = max(peaks)
        result['cuda_device'] = torch.cuda.get_device_name()
        result['cuda_runtime_version'] = torch.version.cuda
        result['cuda_total_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory
    if torch is not None:
        result['torch_num_threads'] = torch.get_num_threads()
        result['torch_matmul_precision'] = torch.get_float32_matmul_precision()
        result['torch_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--snapshots', type=int, default=160, help='Number of training snapshots')
    parser.add_argument('--holdout', type=int, default=20)
    parser.add_argument('--rank', type=int, default=12)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--backends', nargs='+', choices=['numpy', 'torch-cpu', 'torch-cuda'], default=['numpy'])
    parser.add_argument('--pydmd', action='store_true', help='Include optional PyDMD exact-DMD baseline')
    parser.add_argument('--output', type=Path, help='JSON result file; also prints JSON to stdout')
    args = parser.parse_args()
    if args.rank < 1 or args.features < args.rank or args.snapshots <= args.rank + 1:
        parser.error('Require rank >= 1, features >= rank, snapshots > rank + 1')
    if args.holdout < 1 or args.repeats < 1 or args.warmups < 0:
        parser.error('Require holdout >= 1, repeats >= 1, warmups >= 0')
    all_data = synthetic_data(args.features, args.snapshots + args.holdout, args.rank, args.seed, args.dtype)
    training = np.ascontiguousarray(all_data[:, :args.snapshots])
    holdout = all_data[:, args.snapshots:]
    names = list(dict.fromkeys(args.backends + (['pydmd'] if args.pydmd else [])))
    results = []
    for name in names:
        try:
            results.append(benchmark_backend(name, training, holdout, args))
        except Exception as exc:
            results.append({'backend': name, 'status': 'error', 'reason': f'{type(exc).__name__}: {exc}'})
    report = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'cpu': cpu_model(),
            'versions': {name: version(name) for name in ['modaldyn', 'numpy', 'torch', 'pydmd']},
            'thread_settings': {name: os.environ.get(name) for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
        },
        'methodology': {
            'algorithm': 'exact DMD, fixed rank, first-snapshot amplitudes',
            'data': 'synthetic stable real low-rank oscillations; contiguous training / held-out future split',
            'timed': 'fit only, including validation and any internal diagnostics; construction and explicit input transfer excluded',
            'gpu_timing': 'synchronized before and after each measured fit',
            'comparability': 'Implementation work differs, including diagnostics and internal CPU/GPU transfers; no universal speed claim',
            'forecast_alignment': 'first forecast column is one step after last training snapshot',
            'forecast_anchoring': 'All backends project the last observed training state onto fitted modes; PyDMD rollout uses its public modes/eigs explicitly',
            'cuda_memory': 'peak allocated PyTorch bytes, including resident input; not total device memory',
        },
        'results': results,
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n')
    print(encoded)
    return int(any(item['status'] == 'error' for item in results))


if __name__ == '__main__':
    sys.exit(main())
