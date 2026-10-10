from __future__ import annotations

import torch


def newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Muon-style quintic polar iteration in float32."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(int(steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


def inverse_metric_power_2x2(
    metric: torch.Tensor,
    *,
    power: float = 1.0,
    eps: float = 1e-5,
    condition_cap: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable M^(-power/2) for batches of symmetric 2x2 PSD metrics."""
    m = metric.float()
    if m.shape[-2:] != (2, 2):
        raise ValueError(f"metric must end in 2x2, got {tuple(m.shape)}")
    m = 0.5 * (m + m.transpose(-1, -2))
    finite = torch.isfinite(m).all(dim=(-2, -1))
    m = torch.where(finite[..., None, None], m, torch.zeros_like(m))

    a = m[..., 0, 0]
    b = m[..., 0, 1]
    d = m[..., 1, 1]
    scale = torch.maximum(a.abs() + b.abs(), d.abs() + b.abs()).clamp_min(eps)
    an, bn, dn = a / scale, b / scale, d / scale
    centre = 0.5 * (an + dn)
    radius = torch.hypot(0.5 * (an - dn), bn)
    lo_raw = centre - radius
    hi_raw = centre + radius
    hi = hi_raw.clamp_min(eps)
    lo = lo_raw.clamp_min(eps)
    lo = torch.maximum(lo, hi / max(float(condition_cap), 1.0))
    condition = hi / lo.clamp_min(eps)

    scale_factor = scale.pow(-0.5 * power)
    f_lo = lo.pow(-0.5 * power) * scale_factor
    f_hi = hi.pow(-0.5 * power) * scale_factor

    eye = torch.eye(2, device=m.device, dtype=m.dtype).expand(m.shape[:-2] + (2, 2))
    gap = hi_raw - lo_raw
    tolerance = 64.0 * torch.finfo(m.dtype).eps
    projector_hi = m / scale[..., None, None] - lo_raw[..., None, None] * eye
    projector_hi = projector_hi / gap.clamp_min(tolerance)[..., None, None]
    transform = f_lo[..., None, None] * eye + (
        f_hi - f_lo
    )[..., None, None] * projector_hi
    isotropic = (0.5 * (f_lo + f_hi))[..., None, None] * eye
    transform = torch.where((gap <= tolerance)[..., None, None], isotropic, transform)
    transform = torch.where(finite[..., None, None], transform, eye)
    condition = torch.where(finite, condition, torch.ones_like(condition))
    return transform, condition


def inverse_sqrt_diagonal(
    metric: torch.Tensor,
    *,
    power: float = 1.0,
    eps: float = 1e-6,
    condition_cap: float = 100.0,
) -> torch.Tensor:
    """Return a bounded inverse metric power for a diagonal non-negative metric."""
    m = torch.nan_to_num(metric.float(), nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)
    max_value = m.max().clamp_min(eps)
    floor = max_value / max(float(condition_cap), 1.0)
    m = torch.maximum(m, floor)
    return m.pow(-0.5 * float(power))


def restore_joint_frobenius(
    originals: tuple[torch.Tensor, ...], transformed: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    """Restore the aggregate Frobenius norm across one functional family."""
    if len(originals) != len(transformed):
        raise ValueError("original/transformed family sizes differ")
    before_sq = sum(x.float().square().sum() for x in originals)
    after_sq = sum(x.float().square().sum() for x in transformed)
    scale = torch.sqrt(before_sq.clamp_min(1e-24) / after_sq.clamp_min(1e-24))
    return tuple(x * scale.to(x.dtype) for x in transformed)


def silu_prime(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


def swiglu_sensitivities(
    gate_pre: torch.Tensor, up_pre: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel local sensitivities for z = SiLU(g) * u.

    Returns metrics for the up projection, gate projection, and down-projection input.
    """
    gate = torch.nn.functional.silu(gate_pre.float())
    up = up_pre.float()
    up_metric = gate.square().mean(dim=tuple(range(gate.ndim - 1)))
    gate_metric = (up * silu_prime(gate_pre.float())).square().mean(
        dim=tuple(range(gate.ndim - 1))
    )
    down_input_metric = (gate * up).square().mean(dim=tuple(range(gate.ndim - 1)))
    return up_metric, gate_metric, down_input_metric


def categorical_fisher_diagonal(probabilities: torch.Tensor) -> torch.Tensor:
    """Mean diagonal of the categorical Fisher in logit coordinates."""
    p = probabilities.float()
    return (p * (1.0 - p)).mean(dim=tuple(range(p.ndim - 1)))


def causal_delta_second_moment(length: int) -> float:
    """E[(i-j)^2] over all causal pairs j<=i for a length-T sequence."""
    if length <= 1:
        return 0.0
    t = float(length)
    sum_d2 = (t - 1.0) * t * (2.0 * t - 1.0) / 6.0
    sum_d3 = ((t - 1.0) * t / 2.0) ** 2
    numerator = t * sum_d2 - sum_d3
    denominator = t * (t + 1.0) / 2.0
    return numerator / max(denominator, 1.0)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def human_parameters(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
