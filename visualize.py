from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data import make_spirals
from engine import Value
from nn import MLP
from train import label_to_target, mse_loss


def predict(model, x1, x2):
    return model([Value(float(x1)), Value(float(x2))]).data


def evaluate_grid(model, X, resolution=100, padding=0.5):
    """Sample model outputs on a 2D grid; uses Value forward passes only."""
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)

    zz = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            zz[i, j] = predict(model, xx[i, j], yy[i, j])

    return xx, yy, zz


def plot_decision_boundary(
    model,
    X,
    y,
    save_path="decision_boundary.png",
    resolution=100,
    padding=0.5,
    title=None,
    show=False,
):
    """Plot class regions (contour) with spiral data overlaid."""
    xx, yy, zz = evaluate_grid(model, X, resolution=resolution, padding=padding)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(
        xx,
        yy,
        zz,
        levels=[-1.0, 0.0, 1.0],
        colors=["#4393c3", "#fdae61"],
        alpha=0.45,
    )
    ax.contour(xx, yy, zz, levels=[0.0], colors="black", linewidths=0.8)

    for label, color in ((0, "#2166ac"), (1, "#d6604d")):
        mask = y == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            edgecolors="white",
            linewidths=0.4,
            s=24,
            label=f"class {label}",
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def frames_to_gif(frame_paths, gif_path="training.gif", duration_ms=200):
    frames = [Image.open(path) for path in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return gif_path


def train_with_snapshots(
    n_points=200,
    noise=0.3,
    hidden_sizes=(16, 16),
    epochs=400,
    learning_rate=0.01,
    snapshot_every=20,
    frames_dir="frames",
    gif_path="training.gif",
    resolution=60,
    seed=42,
):
    np.random.seed(seed)
    X, y = make_spirals(n_points=n_points, noise=noise)
    model = MLP(2, list(hidden_sizes) + [1])
    params = model.parameters()

    frames_path = Path(frames_dir)
    frames_path.mkdir(parents=True, exist_ok=True)
    frame_files = []

    for epoch in range(1, epochs + 1):
        for xi, yi in zip(X, y):
            x = [Value(float(xi[0])), Value(float(xi[1]))]
            pred = model(x)
            loss = mse_loss(pred, label_to_target(yi))
            loss.backward()
            for p in params:
                p.data -= learning_rate * p.grad
                p.grad = 0.0

        if epoch == 1 or epoch % snapshot_every == 0 or epoch == epochs:
            frame_path = frames_path / f"epoch_{epoch:04d}.png"
            plot_decision_boundary(
                model,
                X,
                y,
                save_path=str(frame_path),
                resolution=resolution,
                title=f"epoch {epoch}",
            )
            frame_files.append(frame_path)

    frames_to_gif(frame_files, gif_path=gif_path)
    plot_decision_boundary(model, X, y, save_path="decision_boundary.png", resolution=100)
    return model, X, y, frame_files, gif_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize MLP decision boundaries.")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Save snapshots during training and build training.gif",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--snapshot-every", type=int, default=20)
    args = parser.parse_args()

    if args.animate:
        train_with_snapshots(
            epochs=args.epochs,
            snapshot_every=args.snapshot_every,
        )
        print("Saved decision_boundary.png and training.gif")
    else:
        from train import train

        model, X, y, _ = train(epochs=args.epochs)
        plot_decision_boundary(model, X, y, save_path="decision_boundary.png")
        print("Saved decision_boundary.png")
