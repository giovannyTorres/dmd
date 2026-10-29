"""Recover a controlled linear system and forecast with known future inputs."""

import numpy as np

from modaldyn import DMDc


def main():
    rng = np.random.default_rng(12)
    a = np.array([[0.93, 0.08], [-0.05, 0.97]])
    b = np.array([[0.15], [0.4]])
    controls = rng.standard_normal((1, 120))
    snapshots = np.empty((2, 121))
    snapshots[:, 0] = [0.4, -0.2]
    for step in range(controls.shape[1]):
        snapshots[:, step + 1] = a @ snapshots[:, step] + b @ controls[:, step]

    # Eighty transition pairs use 81 states and exactly 80 control columns.
    model = DMDc().fit(snapshots[:, :81], controls[:, :80])
    future = model.forecast(controls[:, 80:])
    expected = snapshots[:, 81:]
    error = np.linalg.norm(future - expected) / np.linalg.norm(expected)
    print(f'Operator A error: {np.linalg.norm(model.A_ - a):.3e}')
    print(f'Operator B error: {np.linalg.norm(model.B_ - b):.3e}')
    print(f'Controlled 40-step holdout relative error: {error:.3e}')


if __name__ == '__main__':
    main()
