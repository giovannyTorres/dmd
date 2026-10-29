"""Exact DMD with chronological held-out forecast and optional CUDA.

python examples/quickstart.py
python examples/quickstart.py --backend torch --device cuda
"""

import argparse
import json

import numpy as np

from modaldyn import DMD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='numpy')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    theta = 0.16
    operator = 0.995 * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    snapshots = np.empty((2, 100))
    snapshots[:, 0] = [1.0, 0.2]
    for step in range(1, snapshots.shape[1]):
        snapshots[:, step] = operator @ snapshots[:, step - 1]
    model = DMD(rank=2, dt=0.1, backend=args.backend, device=args.device).fit(snapshots[:, :80])
    prediction = model.forecast(20)
    if hasattr(prediction, 'detach'):
        prediction = prediction.detach().cpu().numpy()
    error = np.linalg.norm(prediction - snapshots[:, 80:]) / np.linalg.norm(snapshots[:, 80:])
    print(f'Held-out 20-step relative error: {error:.3e}')
    print(json.dumps(model.diagnostics(), indent=2))


if __name__ == '__main__':
    main()
