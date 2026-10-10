from __future__ import annotations

from collections.abc import Iterable

import torch

from .geometry import (
    inverse_metric_power_2x2,
    inverse_sqrt_diagonal,
    newton_schulz,
    restore_joint_frobenius,
)


MODE_FAMILIES: dict[str, frozenset[str]] = {
    "muon": frozenset(),
    "orbit_qk": frozenset({"qk"}),
    "orbit_qk_vo": frozenset({"qk", "vo"}),
    "orbit_qk_vo_swiglu": frozenset({"qk", "vo", "swiglu"}),
    "orbit_all": frozenset({"qk", "vo", "swiglu", "embedding", "rope", "moe"}),
    "orbit_all_minus_qk": frozenset({"vo", "swiglu", "embedding", "rope", "moe"}),
    "orbit_all_minus_vo": frozenset({"qk", "swiglu", "embedding", "rope", "moe"}),
    "orbit_all_minus_swiglu": frozenset({"qk", "vo", "embedding", "rope", "moe"}),
    "orbit_all_minus_embedding": frozenset({"qk", "vo", "swiglu", "rope", "moe"}),
}


class OrbitAll(torch.optim.Optimizer):
    """Muon base update plus role-specific function-aware geometry.

    The optimizer intentionally specializes only major functional parameter families.
    Biases, RMSNorm vectors and unrelated scalars remain on an auxiliary AdamW path.
    """

    def __init__(
        self,
        model,
        *,
        mode: str = "orbit_all",
        matrix_lr: float = 0.02,
        aux_lr: float = 0.009,
        weight_decay: float = 0.005,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        functional_power: float = 1.0,
        metric_eps: float = 1e-5,
        metric_condition_cap: float = 100.0,
        deltas: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        families: Iterable[str] | None = None,
    ) -> None:
        if mode not in MODE_FAMILIES and families is None:
            raise ValueError(f"unknown ORBIT-all mode {mode!r}")
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
                ns_steps=int(ns_steps),
                betas=tuple(float(x) for x in adamw_betas),
                eps=float(eps),
            ),
        )
        self.model = model
        self.mode = mode
        self.families = frozenset(families) if families is not None else MODE_FAMILIES[mode]
        unknown = self.families - {"qk", "vo", "swiglu", "embedding", "rope", "moe"}
        if unknown:
            raise ValueError(f"unknown functional families: {sorted(unknown)}")
        self.functional_power = float(functional_power)
        self.metric_eps = float(metric_eps)
        self.metric_condition_cap = float(metric_condition_cap)
        self.deltas = tuple(int(d) for d in deltas)
        self._roles = model.functional_roles()
        self._meta: dict[int, str] = {}
        self._diagnostic_sum: dict[str, float] = {}
        self._diagnostic_count: dict[str, int] = {}
        self._index_roles(params)
        model.set_stat_collection(self.families)

    def _index_roles(self, params: list[torch.nn.Parameter]) -> None:
        for rec in self._roles["embedding"]:
            self._meta[id(rec["weight"])] = "embedding"
        for rec in self._roles["qk"]:
            self._meta[id(rec["q"])] = "q"
            self._meta[id(rec["k"])] = "k"
        for rec in self._roles["vo"]:
            self._meta[id(rec["v"])] = "v"
            self._meta[id(rec["o"])] = "o"
        for rec in self._roles["swiglu"]:
            self._meta[id(rec["gate"])] = "gate"
            self._meta[id(rec["up"])] = "up"
            self._meta[id(rec["down"])] = "down"
        for rec in self._roles["moe"]:
            self._meta[id(rec["router"])] = "router"
        for rec in self._roles["rope"]:
            self._meta[id(rec["frequency"])] = "rope"

        for param in params:
            if id(param) in self._meta:
                continue
            self._meta[id(param)] = "spectral" if param.ndim == 2 else "aux_nodecay"

    def _record(self, key: str, value: torch.Tensor | float) -> None:
        number = float(value.detach().float().mean().cpu()) if torch.is_tensor(value) else float(value)
        self._diagnostic_sum[key] = self._diagnostic_sum.get(key, 0.0) + number
        self._diagnostic_count[key] = self._diagnostic_count.get(key, 0) + 1

    def _spectral_candidate(self, param: torch.Tensor, group: dict) -> torch.Tensor:
        grad = param.grad
        if grad is None:
            raise RuntimeError("spectral candidate requested without gradient")
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        momentum = state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
        lookahead = grad.add(momentum, alpha=group["momentum"])
        update = newton_schulz(lookahead, group["ns_steps"])
        update.mul_(max(1.0, update.size(0) / update.size(1)) ** 0.5)
        return update.to(param.dtype)

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
        step = state["adam_step"]
        exp_avg = state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq = state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        numerator = exp_avg / (1.0 - beta1**step)
        denominator = (exp_avg_sq / (1.0 - beta2**step)).sqrt().add_(group["eps"])
        return numerator / denominator

    def _diag_scale(self, metric: torch.Tensor) -> torch.Tensor:
        return inverse_sqrt_diagonal(
            metric,
            power=self.functional_power,
            eps=self.metric_eps,
            condition_cap=self.metric_condition_cap,
        )

    @torch.no_grad()
    def _apply_qk(self, candidates: dict[int, torch.Tensor]) -> None:
        if "qk" not in self.families:
            return
        for rec in self._roles["qk"]:
            q, k, module = rec["q"], rec["k"], rec["module"]
            if id(q) not in candidates or id(k) not in candidates:
                continue
            q0, k0 = candidates[id(q)], candidates[id(k)]
            mq, mk = module.qk_metrics(self.deltas, rotate=True, eps=self.metric_eps)
            tq, cq = inverse_metric_power_2x2(
                mq.to(q0.device),
                power=self.functional_power,
                eps=self.metric_eps,
                condition_cap=self.metric_condition_cap,
            )
            tk, ck = inverse_metric_power_2x2(
                mk.to(k0.device),
                power=self.functional_power,
                eps=self.metric_eps,
                condition_cap=self.metric_condition_cap,
            )
            qv = q0.float().view(module.n_head, module.n_freq, 2, -1)
            kv = k0.float().view(module.n_kv_head, module.n_freq, 2, -1)
            q1 = torch.einsum("hfab,hfbi->hfai", tq, qv).reshape_as(q0)
            k1 = torch.einsum("hfab,hfbi->hfai", tk, kv).reshape_as(k0)
            q1, k1 = restore_joint_frobenius((q0, k0), (q1, k1))
            candidates[id(q)], candidates[id(k)] = q1.to(q0.dtype), k1.to(k0.dtype)
            self._record("qk_condition", torch.cat((cq.flatten(), ck.flatten())).mean())

    @torch.no_grad()
    def _apply_vo(self, candidates: dict[int, torch.Tensor]) -> None:
        if "vo" not in self.families:
            return
        for rec in self._roles["vo"]:
            v, o, module = rec["v"], rec["o"], rec["module"]
            if id(v) not in candidates or id(o) not in candidates:
                continue
            v0, o0 = candidates[id(v)], candidates[id(o)]
            v_metric, o_metric = module.vo_metrics(self.metric_eps)
            v_scale = self._diag_scale(v_metric.to(v0.device))
            o_scale = self._diag_scale(o_metric.to(o0.device))
            v1 = v0.float().view(module.n_kv_head, module.head_dim, -1)
            v1 = (v1 * v_scale[..., None]).reshape_as(v0)
            o1 = o0.float().view(o0.size(0), module.n_head, module.head_dim)
            o1 = (o1 * o_scale[None, ...]).reshape_as(o0)
            v1, o1 = restore_joint_frobenius((v0, o0), (v1, o1))
            candidates[id(v)], candidates[id(o)] = v1.to(v0.dtype), o1.to(o0.dtype)

    @torch.no_grad()
    def _apply_swiglu(self, candidates: dict[int, torch.Tensor]) -> None:
        if "swiglu" not in self.families:
            return
        for rec in self._roles["swiglu"]:
            gate, up, down, module = rec["gate"], rec["up"], rec["down"], rec["module"]
            ids = (id(gate), id(up), id(down))
            if any(i not in candidates for i in ids):
                continue
            g0, u0, d0 = (candidates[i] for i in ids)
            gate_scale = self._diag_scale(module.gate_metric.to(g0.device))
            up_scale = self._diag_scale(module.up_metric.to(u0.device))
            down_scale = self._diag_scale(module.down_metric.to(d0.device))
            g1 = g0.float() * gate_scale[:, None]
            u1 = u0.float() * up_scale[:, None]
            d1 = d0.float() * down_scale[None, :]
            g1, u1, d1 = restore_joint_frobenius((g0, u0, d0), (g1, u1, d1))
            candidates[id(gate)] = g1.to(g0.dtype)
            candidates[id(up)] = u1.to(u0.dtype)
            candidates[id(down)] = d1.to(d0.dtype)

    @torch.no_grad()
    def _apply_embedding(self, candidates: dict[int, torch.Tensor]) -> None:
        if "embedding" not in self.families:
            return
        for rec in self._roles["embedding"]:
            weight, module = rec["weight"], rec["module"]
            if id(weight) not in candidates:
                continue
            update = candidates[id(weight)]
            scale = self._diag_scale(module.token_frequency.to(update.device))
            transformed = update.float() * scale[:, None]
            (transformed,) = restore_joint_frobenius((update,), (transformed,))
            candidates[id(weight)] = transformed.to(update.dtype)

    @torch.no_grad()
    def _apply_rope(self, candidates: dict[int, torch.Tensor]) -> None:
        if "rope" not in self.families:
            return
        for rec in self._roles["rope"]:
            frequency, module = rec["frequency"], rec["module"]
            if id(frequency) not in candidates:
                continue
            update = candidates[id(frequency)]
            metric = module.rope_metric(self.deltas, eps=self.metric_eps).to(update.device)
            scale = self._diag_scale(metric)
            transformed = update.float() * scale
            (transformed,) = restore_joint_frobenius((update,), (transformed,))
            candidates[id(frequency)] = transformed.to(update.dtype)

    @torch.no_grad()
    def _apply_moe(self, candidates: dict[int, torch.Tensor]) -> None:
        if "moe" not in self.families:
            return
        for rec in self._roles["moe"]:
            router, module = rec["router"], rec["module"]
            if id(router) not in candidates:
                continue
            update = candidates[id(router)]
            scale = self._diag_scale(module.router_fisher.to(update.device))
            transformed = update.float() * scale[:, None]
            (transformed,) = restore_joint_frobenius((update,), (transformed,))
            candidates[id(router)] = transformed.to(update.dtype)
            self._record("moe_max_load", module.expert_load.max())

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None if closure is None else closure()
        group = self.param_groups[0]
        candidates: dict[int, torch.Tensor] = {}

        adam_roles = {"embedding", "rope", "aux_nodecay"}
        for param in group["params"]:
            if param.grad is None:
                continue
            role = self._meta[id(param)]
            if role in adam_roles:
                candidates[id(param)] = self._adam_candidate(param, group)
            else:
                candidates[id(param)] = self._spectral_candidate(param, group)

        self._apply_qk(candidates)
        self._apply_vo(candidates)
        self._apply_swiglu(candidates)
        self._apply_embedding(candidates)
        self._apply_rope(candidates)
        self._apply_moe(candidates)

        matrix_roles = {"q", "k", "v", "o", "gate", "up", "down", "router", "spectral"}
        aux_lr = float(group["lr"]) * float(group["aux_lr_ratio"])
        for param in group["params"]:
            if param.grad is None:
                continue
            role = self._meta[id(param)]
            lr = float(group["lr"]) if role in matrix_roles else aux_lr
            decay = role not in {"aux_nodecay", "rope"}
            if decay and group["weight_decay"]:
                param.mul_(1.0 - lr * float(group["weight_decay"]))
            param.add_(candidates[id(param)], alpha=-lr)
        return loss

    def auxiliary_lr(self) -> float:
        group = self.param_groups[0]
        return float(group["lr"] * group["aux_lr_ratio"])

    def diagnostics(self, *, reset: bool = False) -> dict[str, float]:
        values = {
            key: self._diagnostic_sum[key] / max(self._diagnostic_count.get(key, 1), 1)
            for key in sorted(self._diagnostic_sum)
        }
        if reset:
            self._diagnostic_sum.clear()
            self._diagnostic_count.clear()
        return values
