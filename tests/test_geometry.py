import torch

from orbit_all.geometry import (
    inverse_metric_power_2x2,
    inverse_sqrt_diagonal,
    restore_joint_frobenius,
)


def test_inverse_metric_power_isotropic():
    metric = 4.0 * torch.eye(2).view(1, 1, 2, 2)
    transform, condition = inverse_metric_power_2x2(metric, power=1.0)
    expected = 0.5 * torch.eye(2).view(1, 1, 2, 2)
    assert torch.allclose(transform, expected, atol=1e-5)
    assert torch.allclose(condition, torch.ones_like(condition))


def test_diagonal_condition_cap():
    metric = torch.tensor([1e-12, 1.0])
    scale = inverse_sqrt_diagonal(metric, condition_cap=100.0)
    assert torch.isfinite(scale).all()
    assert float(scale.max() / scale.min()) <= 10.0001


def test_joint_restore_preserves_family_norm():
    a = torch.randn(7, 5)
    b = torch.randn(3, 5)
    a2, b2 = restore_joint_frobenius((a, b), (3.0 * a, 0.2 * b))
    before = a.square().sum() + b.square().sum()
    after = a2.square().sum() + b2.square().sum()
    assert torch.allclose(before, after, rtol=1e-5, atol=1e-6)
