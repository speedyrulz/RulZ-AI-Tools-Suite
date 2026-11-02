"""Gradient norm logging: the step's norm before clipping, the clip itself unchanged, and what the log and the
status file show."""
import torch

from rulztrainer.trainer import format_grad_norm, round_norms, step_grad_norm


def params_with_grads(*grads):
    out = []
    for g in grads:
        p = torch.nn.Parameter(torch.zeros_like(g))
        p.grad = g.clone()
        out.append(p)
    return out


def test_the_norm_is_measured_before_the_clip():
    params = params_with_grads(torch.tensor([3.0, 0.0]), torch.tensor([0.0, 4.0]))
    assert step_grad_norm(params, 1.0) == 5.0
    assert torch.allclose(torch.cat([p.grad for p in params]).norm(), torch.tensor(1.0))  # clipped as before


def test_no_clip_leaves_the_gradients_alone():
    params = params_with_grads(torch.tensor([3.0, 4.0]))
    assert step_grad_norm(params, 0.0) == 5.0
    assert torch.equal(params[0].grad, torch.tensor([3.0, 4.0]))
    assert step_grad_norm([torch.nn.Parameter(torch.zeros(2))], 0.0) is None  # no gradients at all


def test_a_norm_that_is_not_finite_is_none():
    assert step_grad_norm(params_with_grads(torch.tensor([float("nan"), 1.0])), 0.0) is None
    assert step_grad_norm(params_with_grads(torch.tensor([float("inf"), 1.0])), 1.0) is None


def test_log_and_status_text():
    assert format_grad_norm(0.4123, 1.0) == "0.412"
    assert format_grad_norm(3.2, 1.0) == "3.2 (clipped)"
    assert format_grad_norm(3.2, 0.0) == "3.2"
    assert format_grad_norm(None, 1.0) == "nan"
    assert round_norms([0.123456, None, 12345.678]) == [0.1235, None, 12350.0]
