import numpy as np

from data import make_spirals
from engine import Value
from nn import MLP


def label_to_target(label):
    """Map class labels {0, 1} to {-1, 1} for tanh output neurons."""
    return 1.0 if label == 1 else -1.0


def mse_loss(pred, target):
    return (pred - target) ** 2


def accuracy(model, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        pred = model([Value(float(xi[0])), Value(float(xi[1]))])
        predicted = 1 if pred.data > 0 else 0
        correct += int(predicted == yi)
    return correct / len(y)


def train(
    n_points=200,
    noise=0.3,
    hidden_sizes=(16, 16),
    epochs=400,
    learning_rate=0.01,
    log_every=20,
    seed=42,
):
    np.random.seed(seed)
    X, y = make_spirals(n_points=n_points, noise=noise)
    model = MLP(2, list(hidden_sizes) + [1])
    params = model.parameters()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for xi, yi in zip(X, y):
            x = [Value(float(xi[0])), Value(float(xi[1]))]
            pred = model(x)
            loss = mse_loss(pred, label_to_target(yi))
            total_loss += loss.data

            loss.backward()

            for p in params:
                p.data -= learning_rate * p.grad
                p.grad = 0.0

        if epoch % log_every == 0 or epoch == epochs:
            avg_loss = total_loss / len(y)
            acc = accuracy(model, X, y)
            print(f"epoch {epoch:4d}  loss {avg_loss:.4f}  accuracy {acc:.3f}")

    final_acc = accuracy(model, X, y)
    print(f"\nFinal accuracy: {final_acc:.3f}")
    return model, X, y, final_acc


if __name__ == "__main__":
    train()
