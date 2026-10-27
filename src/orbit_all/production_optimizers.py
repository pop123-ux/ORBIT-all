from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from .geometry import newton_schulz
from .optimizer import OrbitAll


def _split_decay_params(model, weight_decay: float) -> list[dict]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class Lion(torch.optim.Optimizer):
    """Decoupled Lion optimizer used as a production-family baseline."""

    def __init__(
        self,
        params: Iterable,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        beta1, beta2 = (float(x) for x in betas)
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("betas must lie in [0, 1)")
        super().__init__(
            params,
            dict(lr=float(lr), betas=(beta1, beta2), weight_decay=float(weight_decay)),
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                param.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss


class PaLMAdafactor(torch.optim.Optimizer):
    """PaLM-style unfactorized Adafactor with RMS parameter scaling.

    This keeps the optimizer mechanics disclosed by PaLM while the benchmark
    deliberately uses one common schedule across optimizer families.
    """

    def __init__(
        self,
        params: Iterable,
        *,
        lr: float,
        beta1: float = 0.9,
        beta2_power: float = 0.8,
        eps: float = 1e-30,
        parameter_scale_floor: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        super().__init__(
            params,
            dict(
                lr=float(lr),
                beta1=float(beta1),
                beta2_power=float(beta2_power),
                eps=float(eps),
                parameter_scale_floor=float(parameter_scale_floor),
                weight_decay=float(weight_decay),
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            base_lr = float(group["lr"])
            beta1 = float(group["beta1"])
            beta2_power = float(group["beta2_power"])
            eps = float(group["eps"])
            scale_floor = float(group["parameter_scale_floor"])
            weight_decay = float(group["weight_decay"])
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                step = int(state["step"])
                beta2 = 1.0 - float(step) ** (-beta2_power)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                update = exp_avg / exp_avg_sq.sqrt().add_(eps)
                rms = param.float().square().mean().sqrt().clamp_min(scale_floor)
                lr = base_lr * float(rms)
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                param.add_(update, alpha=-lr)
        return loss


class MoonshotMuon(torch.optim.Optimizer):
    """Moonlight/Kimi-style Muon with optional Kimi K2 QK-Clip.

    Matrix parameters use momentum, Newton-Schulz orthogonalization, and
    Moonshot's 0.2*sqrt(max(m,n)) Adam-RMS matching. Embeddings and non-matrix
    parameters use an auxiliary AdamW-style update. For MuonClip, forward hooks
    collect per-head causal max logits and post-update Q/K clipping is applied.
    """

    def __init__(
        self,
        model,
        *,
        matrix_lr: float,
        aux_lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        rms_scale: float = 0.2,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        qk_clip: bool = False,
        qk_clip_threshold: float = 100.0,
        logit_chunk_size: int = 128,
    ) -> None:
        if matrix_lr <= 0 or aux_lr < 0:
            raise ValueError("matrix_lr must be positive and aux_lr non-negative")
        params = list(model.parameters())
        super().__init__(
            params,
            dict(
                lr=float(matrix_lr),
                aux_lr_ratio=float(aux_lr) / float(matrix_lr),
                weight_decay=float(weight_decay),
                momentum=float(momentum),
                nesterov=bool(nesterov),
                ns_steps=int(ns_steps),
                rms_scale=float(rms_scale),
                betas=tuple(float(x) for x in adamw_betas),
                eps=float(eps),
            ),
        )
        self.model = model
        self.families: frozenset[str] = frozenset()
        self.qk_clip = bool(qk_clip)
        self.qk_clip_threshold = float(qk_clip_threshold)
        self.logit_chunk_size = int(logit_chunk_size)
        self._roles = model.functional_roles()
        embedding_ids = {id(rec["weight"]) for rec in self._roles["embedding"]}
        self._matrix_ids = {
            id(param) for param in params if param.ndim == 2 and id(param) not in embedding_ids
        }
        self._diag_sum: dict[str, float] = {}
        self._diag_count: dict[str, int] = {}
        self._max_logits: dict[int, torch.Tensor] = {}
        self._hooks = []
        model.set_stat_collection(())
        if self.qk_clip:
            self._register_qk_hooks()

    def _record(self, key: str, value: float) -> None:
        self._diag_sum[key] = self._diag_sum.get(key, 0.0) + float(value)
        self._diag_count[key] = self._diag_count.get(key, 0) + 1

    def _register_qk_hooks(self) -> None:
        for rec in self._roles["qk"]:
            layer = int(rec["layer"])
            module = rec["module"]

            def hook(mod, args, *, layer_idx=layer):
                if not mod.training:
                    return
                x = args[0]
                with torch.no_grad():
                    q = mod.q_proj(x).view(
                        x.size(0), x.size(1), mod.n_head, mod.head_dim
                    ).transpose(1, 2)
                    k = mod.k_proj(x).view(
                        x.size(0), x.size(1), mod.n_kv_head, mod.head_dim
                    ).transpose(1, 2)
                    if mod.config.qk_norm:
                        q = q * q.float().square().mean(
                            dim=-1, keepdim=True
                        ).add(1e-6).rsqrt().to(q.dtype)
                        k = k * k.float().square().mean(
                            dim=-1, keepdim=True
                        ).add(1e-6).rsqrt().to(k.dtype)
                    q = mod._apply_rope(q)
                    k = mod._apply_rope(k)
                    if mod.group_size > 1:
                        k = k.repeat_interleave(mod.group_size, dim=1)
                    current = self._causal_head_max(q, k, mod.head_dim)
                    previous = self._max_logits.get(layer_idx)
                    self._max_logits[layer_idx] = (
                        current if previous is None else torch.maximum(previous, current)
                    )

            self._hooks.append(module.register_forward_pre_hook(hook))

    def _causal_head_max(
        self, q: torch.Tensor, k: torch.Tensor, head_dim: int
    ) -> torch.Tensor:
        qf = q.float()
        kf = k.float()
        kt = kf.transpose(-1, -2)
        length = q.size(2)
        scale = 1.0 / math.sqrt(float(head_dim))
        best = torch.full((q.size(1),), -torch.inf, device=q.device, dtype=torch.float32)
        positions = torch.arange(length, device=q.device)
        for start in range(0, length, self.logit_chunk_size):
            end = min(start + self.logit_chunk_size, length)
            logits = torch.matmul(qf[:, :, start:end], kt) * scale
            row = positions[start:end, None]
            col = positions[None, :]
            valid = col <= row
            logits = logits.masked_fill(~valid[None, None], -torch.inf)
            best = torch.maximum(best, logits.amax(dim=(0, 2, 3)))
        return best

    def _adam_candidate(self, param: torch.Tensor, group: dict) -> torch.Tensor:
        grad = param.grad
        if grad is None:
            raise RuntimeError("Adam candidate requested without gradient")
        beta1, beta2 = group["betas"]
        state = self.state[param]
        if "adam_step" not in state:
            state["adam_step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        state["adam_step"] += 1
        step = int(state["adam_step"])
        exp_avg = state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq = state["exp_avg_sq"].mul_(beta2).addcmul_(
            grad, grad, value=1.0 - beta2
        )
        numerator = exp_avg / (1.0 - beta1**step)
        denominator = (exp_avg_sq / (1.0 - beta2**step)).sqrt().add_(group["eps"])
        return numerator / denominator

    @torch.no_grad()
    def _apply_qk_clip(self) -> None:
        if not self.qk_clip:
            return
        clipped_heads = 0
        total_heads = 0
        observed_max = 0.0
        for rec in self._roles["qk"]:
            layer = int(rec["layer"])
            max_logits = self._max_logits.get(layer)
            if max_logits is None:
                continue
            q = rec["q"]
            k = rec["k"]
            module = rec["module"]
            finite = torch.nan_to_num(
                max_logits.float(),
                nan=0.0,
                posinf=self.qk_clip_threshold * 10.0,
                neginf=0.0,
            )
            observed_max = max(observed_max, float(finite.max().cpu()))
            gamma = torch.clamp(
                self.qk_clip_threshold / finite.clamp_min(1e-12),
                max=1.0,
            )
            total_heads += int(gamma.numel())
            clipped_heads += int((gamma < 1.0).sum().cpu())
            q_scale = gamma.sqrt().to(q.device, dtype=q.dtype)
            q.view(module.n_head, module.head_dim, -1).mul_(q_scale[:, None, None])
            if module.n_kv_head == module.n_head:
                k_scale = q_scale
            else:
                grouped = gamma.view(module.n_kv_head, module.group_size)
                k_scale = grouped.amin(dim=1).sqrt().to(k.device, dtype=k.dtype)
            k.view(module.n_kv_head, module.head_dim, -1).mul_(k_scale[:, None, None])
        if total_heads:
            self._record("qk_clip_fraction", clipped_heads / total_heads)
            self._record("qk_max_logit", observed_max)
        self._max_logits.clear()

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None if closure is None else closure()
        group = self.param_groups[0]
        matrix_lr = float(group["lr"])
        aux_lr = matrix_lr * float(group["aux_lr_ratio"])
        weight_decay = float(group["weight_decay"])
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if id(param) in self._matrix_ids:
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum = state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                candidate = (
                    grad.add(momentum, alpha=group["momentum"])
                    if group["nesterov"]
                    else momentum
                )
                update = newton_schulz(candidate, group["ns_steps"])
                scale = float(group["rms_scale"]) * math.sqrt(
                    float(max(param.shape[0], param.shape[1]))
                )
                update = update.to(param.dtype).mul_(scale)
                if weight_decay:
                    param.mul_(1.0 - matrix_lr * weight_decay)
                param.add_(update, alpha=-matrix_lr)
            else:
                update = self._adam_candidate(param, group)
                if param.ndim >= 2 and weight_decay:
                    param.mul_(1.0 - aux_lr * weight_decay)
                param.add_(update, alpha=-aux_lr)
        self._apply_qk_clip()
        return loss

    def diagnostics(self, *, reset: bool = False) -> dict[str, float]:
        result = {
            key: self._diag_sum[key] / max(1, self._diag_count.get(key, 1))
            for key in sorted(self._diag_sum)
        }
        if reset:
            self._diag_sum.clear()
            self._diag_count.clear()
        return result


def build_production_optimizer(model, name: str, hparams: dict, shared: dict):
    """Construct one optimizer baseline from the production benchmark protocol."""
    if name == "orbit_all":
        return OrbitAll(
            model,
            mode="orbit_all",
            matrix_lr=float(hparams["matrix_lr"]),
            aux_lr=float(hparams["aux_lr"]),
            weight_decay=float(hparams["weight_decay"]),
            momentum=float(shared["momentum"]),
            nesterov=True,
            ns_steps=int(shared["ns_steps"]),
            functional_power=float(shared["functional_power"]),
            metric_eps=float(shared["metric_eps"]),
            metric_condition_cap=float(shared["metric_condition_cap"]),
        )
    if name in {"muon_moonlight", "muonclip_kimi"}:
        return MoonshotMuon(
            model,
            matrix_lr=float(hparams["matrix_lr"]),
            aux_lr=float(hparams["aux_lr"]),
            weight_decay=float(hparams["weight_decay"]),
            momentum=float(shared["momentum"]),
            ns_steps=int(shared["ns_steps"]),
            rms_scale=float(shared["moonshot_rms_scale"]),
            qk_clip=name == "muonclip_kimi",
            qk_clip_threshold=float(shared["qk_clip_threshold"]),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            _split_decay_params(model, float(hparams["weight_decay"])),
            lr=float(hparams["lr"]),
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    if name == "adam":
        return torch.optim.Adam(
            _split_decay_params(model, float(hparams["weight_decay"])),
            lr=float(hparams["lr"]),
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    if name == "lion":
        return Lion(
            _split_decay_params(model, float(hparams["weight_decay"])),
            lr=float(hparams["lr"]),
            betas=(0.9, 0.99),
        )
    if name == "adafactor_palm":
        return PaLMAdafactor(
            _split_decay_params(model, float(hparams["weight_decay"])),
            lr=float(hparams["lr"]),
            beta1=0.9,
            beta2_power=0.8,
            eps=1e-30,
            parameter_scale_floor=1e-3,
        )
    raise ValueError(f"unknown production optimizer {name!r}")


def optimizer_base_lr(name: str, hparams: dict) -> float:
    matrix_family = {"orbit_all", "muon_moonlight", "muonclip_kimi"}
    return float(hparams["matrix_lr"] if name in matrix_family else hparams["lr"])


def set_optimizer_lr(optimizer, name: str, base_lr: float, multiplier: float) -> None:
    lr = float(base_lr) * float(multiplier)
    if name in {"orbit_all", "muon_moonlight", "muonclip_kimi"}:
        optimizer.param_groups[0]["lr"] = lr
        return
    for group in optimizer.param_groups:
        group["lr"] = lr


def optimizer_diagnostics(optimizer) -> dict[str, float]:
    method = getattr(optimizer, "diagnostics", None)
    return {} if method is None else dict(method())
