"""Tests cho src.model.network."""
import torch

from src.model.network import PolicyValueNet, PolicyNet, NUM_ACTIONS


def test_forward_shapes():
    model = PolicyValueNet(channels=32, n_res_blocks=1)  # nhỏ cho test nhanh
    x = torch.randn(2, 12, 8, 8)
    p, v = model(x)
    assert p.shape == (2, NUM_ACTIONS)
    assert v.shape == (2,)
    assert torch.all(v.abs() <= 1.0 + 1e-6)  # tanh output


def test_policy_net_wrapper():
    model = PolicyNet(channels=32, n_res_blocks=1)
    x = torch.randn(2, 12, 8, 8)
    p = model(x)
    assert p.shape == (2, NUM_ACTIONS)


def test_backward():
    """Smoke test: model có thể backprop."""
    model = PolicyValueNet(channels=32, n_res_blocks=1)
    x = torch.randn(4, 12, 8, 8)
    p, v = model(x)
    loss = p.sum() + v.sum()
    loss.backward()
    # Check có gradient
    has_grad = any(param.grad is not None and param.grad.abs().sum() > 0
                   for param in model.parameters())
    assert has_grad
