import torch
import pytest

from engine import Value


def assert_grads_match(values, tensors):
    for value, tensor in zip(values, tensors):
        assert value.grad == pytest.approx(tensor.grad.item())


def run_value_and_torch(build, *inputs):
    values = [Value(x) for x in inputs]
    tensors = [
        torch.tensor(x, requires_grad=True, dtype=torch.float64) for x in inputs
    ]

    out_v = build(values)
    out_t = build(tensors)

    out_v.backward()
    out_t.backward()

    assert_grads_match(values, tensors)


def test_add():
    run_value_and_torch(lambda v: v[0] + v[1], 2.0, 3.0)


def test_mul():
    run_value_and_torch(lambda v: v[0] * v[1], 2.0, -3.0)


def test_sub():
    run_value_and_torch(lambda v: v[0] - v[1], 5.0, 2.0)


def test_div():
    run_value_and_torch(lambda v: v[0] / v[1], 6.0, 2.0)


def test_pow():
    run_value_and_torch(lambda v: v[0] ** 2, 3.0)


def test_mul_add_tanh():
    def build(v):
        return (v[0] * v[1] + v[2]).tanh() if isinstance(v[0], Value) else torch.tanh(
            v[0] * v[1] + v[2]
        )

    run_value_and_torch(build, 2.0, -3.0, 10.0)
