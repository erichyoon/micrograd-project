import matplotlib.pyplot as plt
import numpy as np


def make_spirals(n_points=200, noise=0.3):
    """Two interleaved spirals, labels in {0, 1}."""
    n = n_points // 2
    theta = np.linspace(0, 4 * np.pi, n)
    r = np.linspace(0, 1, n)

    x0 = r * np.cos(theta) + noise * np.random.randn(n)
    y0 = r * np.sin(theta) + noise * np.random.randn(n)

    # Rotate by pi to interleave a second spiral arm.
    x1 = r * np.cos(theta + np.pi) + noise * np.random.randn(n)
    y1 = r * np.sin(theta + np.pi) + noise * np.random.randn(n)

    X = np.column_stack([np.concatenate([x0, x1]), np.concatenate([y0, y1])])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


def plot_spirals(X, y):
    """Scatter plot colored by class."""
    fig, ax = plt.subplots()

    for label in (0, 1):
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            label=f"class {label}",
            alpha=0.7,
            s=20,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    fig.tight_layout()
    return fig, ax
