from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import categorical_fisher_diagonal, silu_prime


@dataclass
class ModelConfig:
    vocab_size: int = 50_257
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 768
    mlp_hidden: int = 2048
    dropout: float = 0.0
    bias: bool = False
    rope_base: float = 10_000.0
    rope_mode: str = "fixed"  # fixed | learned_log_frequency
    qk_norm: bool = False
    stats_beta: float = 0.95
    moe_experts: int = 0
    gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        dim = self.n_embd // self.n_head
        if dim % 2:
            raise ValueError("RoPE head dimension must be even")
        return dim

    def validate(self) -> None:
        _ = self.head_dim
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head for GQA")
        if self.rope_mode not in {"fixed", "learned_log_frequency"}:
            raise ValueError(f"unsupported rope_mode={self.rope_mode!r}")
        if self.moe_experts < 0:
            raise ValueError("moe_experts must be non-negative")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt().to(x.dtype)
        return x * scale * self.weight


def _ema_(target: torch.Tensor, value: torch.Tensor, beta: float, seen: torch.Tensor) -> None:
    effective_beta = float(beta) if int(seen.item()) > 0 else 0.0
    target.mul_(effective_beta).add_(value.to(target), alpha=1.0 - effective_beta)


def _rms_head_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * x.float().square().mean(dim=-1, keepdim=True).add(eps).rsqrt().to(x.dtype)


class FunctionalAttention(nn.Module):
    """Causal RoPE attention with statistics consumed by ORBIT-all.

    Q/K statistics are collected after optional QK normalization and before RoPE.
    V/O statistics summarize content transport. GQA mapping is explicit: each KV head
    owns an equal-size contiguous group of query heads.
    """

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.group_size = self.n_head // self.n_kv_head
        self.head_dim = config.head_dim
        self.n_freq = self.head_dim // 2
        kv_width = self.n_kv_head * self.head_dim

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, kv_width, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, kv_width, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        inv_freq = 1.0 / (
            config.rope_base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        if config.rope_mode == "learned_log_frequency":
            self.log_inv_freq = nn.Parameter(inv_freq.log())
        else:
            self.register_buffer("fixed_inv_freq", inv_freq, persistent=True)
            self.log_inv_freq = None

        q_eye = torch.eye(2, dtype=torch.float32).view(1, 1, 2, 2)
        self.register_buffer(
            "q_second", q_eye.repeat(self.n_head, self.n_freq, 1, 1), persistent=True
        )
        self.register_buffer(
            "k_second", q_eye.repeat(self.n_kv_head, self.n_freq, 1, 1), persistent=True
        )
        self.register_buffer(
            "v_second", torch.ones(self.n_kv_head, self.head_dim), persistent=True
        )
        self.register_buffer(
            "context_second", torch.ones(self.n_head, self.head_dim), persistent=True
        )
        self.register_buffer("stats_seen", torch.zeros((), dtype=torch.long), persistent=True)
        self.collect_stats = False

    def inv_freq(self) -> torch.Tensor:
        if self.log_inv_freq is not None:
            return self.log_inv_freq.exp()
        return self.fixed_inv_freq

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        time = x.size(2)
        inv = self.inv_freq().float().to(x.device)
        angles = torch.arange(time, device=x.device, dtype=torch.float32)[:, None] * inv[None, :]
        cos = angles.cos().to(x.dtype).view(1, 1, time, self.n_freq, 1)
        sin = angles.sin().to(x.dtype).view(1, 1, time, self.n_freq, 1)
        pair = x.view(*x.shape[:-1], self.n_freq, 2)
        a, b = pair[..., :1], pair[..., 1:]
        return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)

    def _rotation(self, delta: int) -> torch.Tensor:
        angle = self.inv_freq().float() * float(delta)
        c, s = angle.cos(), angle.sin()
        return torch.stack(
            (torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2
        )

    @torch.no_grad()
    def _update_pre_attention_stats(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if not self.collect_stats or not self.training:
            return
        q2 = q.float().view(q.size(0), self.n_head, q.size(2), self.n_freq, 2)
        k2 = k.float().view(k.size(0), self.n_kv_head, k.size(2), self.n_freq, 2)
        q_second = torch.einsum("bhtfi,bhtfj->hfij", q2, q2) / max(1, q.size(0) * q.size(2))
        k_second = torch.einsum("bhtfi,bhtfj->hfij", k2, k2) / max(1, k.size(0) * k.size(2))
        v_second = v.float().square().mean(dim=(0, 2))
        _ema_(self.q_second, q_second, self.config.stats_beta, self.stats_seen)
        _ema_(self.k_second, k_second, self.config.stats_beta, self.stats_seen)
        _ema_(self.v_second, v_second, self.config.stats_beta, self.stats_seen)

    @torch.no_grad()
    def _update_context_stats(self, context: torch.Tensor) -> None:
        if not self.collect_stats or not self.training:
            return
        value = context.float().square().mean(dim=(0, 2))
        _ema_(self.context_second, value, self.config.stats_beta, self.stats_seen)
        self.stats_seen.add_(1)

    @torch.no_grad()
    def qk_metrics(
        self, deltas: tuple[int, ...], *, rotate: bool = True, eps: float = 1e-5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_second.float()
        k = self.k_second.float()
        q_groups = q.view(self.n_kv_head, self.group_size, self.n_freq, 2, 2).mean(dim=1)
        k_for_q = k.repeat_interleave(self.group_size, dim=0)
        if not rotate:
            mq, mk = k_for_q.clone(), q_groups.clone()
        else:
            mq = torch.zeros_like(q)
            mk = torch.zeros_like(k)
            ds = tuple(int(d) for d in deltas) or (0,)
            for delta in ds:
                r = self._rotation(delta).to(q.device).unsqueeze(0)
                rt = r.transpose(-1, -2)
                mq.add_(r @ k_for_q @ rt)
                mk.add_(rt @ q_groups @ r)
            mq.div_(len(ds))
            mk.div_(len(ds))
        eye = torch.eye(2, device=q.device, dtype=q.dtype).view(1, 1, 2, 2)
        return mq + eps * eye, mk + eps * eye

    @torch.no_grad()
    def vo_metrics(self, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        context_group = self.context_second.view(
            self.n_kv_head, self.group_size, self.head_dim
        ).mean(dim=1)
        # Ratio is a channelwise proxy for how strongly attention mixing transports V variation.
        v_metric = context_group / self.v_second.float().clamp_min(eps)
        o_metric = self.context_second.float()
        return v_metric.clamp_min(eps), o_metric.clamp_min(eps)

    @torch.no_grad()
    def rope_metric(self, deltas: tuple[int, ...], eps: float = 1e-6) -> torch.Tensor:
        """Approximate score sensitivity for learned log-frequency parameters."""
        q = self.q_second.float()
        k_for_q = self.k_second.float().repeat_interleave(self.group_size, dim=0)
        j = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device=q.device)
        inv = self.inv_freq().float().to(q.device)
        out = torch.zeros(self.n_freq, device=q.device)
        ds = tuple(int(d) for d in deltas) or (0,)
        for delta in ds:
            r = self._rotation(delta).to(q.device).unsqueeze(0)
            jr = j.view(1, 1, 2, 2) @ r
            # E[(q^T J R k)^2] under the local second-moment approximation.
            sensitivity = torch.einsum(
                "hfij,hfjk,hfkl,hfli->hf", q, jr, k_for_q, jr.transpose(-1, -2)
            ).clamp_min(0.0)
            out.add_(sensitivity.mean(dim=0) * (float(delta) * inv).square())
        return out.div_(len(ds)).add_(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.config.qk_norm:
            q = _rms_head_norm(q)
            k = _rms_head_norm(k)
        self._update_pre_attention_stats(q.detach(), k.detach(), v.detach())
        q, k = self._apply_rope(q), self._apply_rope(k)
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=True,
        )
        self._update_context_stats(context.detach())
        merged = context.transpose(1, 2).contiguous().view(bsz, seqlen, self.config.n_embd)
        return self.o_proj(merged)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.n_embd, config.mlp_hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.mlp_hidden, bias=config.bias)
        self.down_proj = nn.Linear(config.mlp_hidden, config.n_embd, bias=config.bias)
        self.register_buffer("up_metric", torch.ones(config.mlp_hidden), persistent=True)
        self.register_buffer("gate_metric", torch.ones(config.mlp_hidden), persistent=True)
        self.register_buffer("down_metric", torch.ones(config.mlp_hidden), persistent=True)
        self.register_buffer("stats_seen", torch.zeros((), dtype=torch.long), persistent=True)
        self.collect_stats = False

    @torch.no_grad()
    def _update_stats(self, gate_pre: torch.Tensor, up_pre: torch.Tensor, z: torch.Tensor) -> None:
        if not self.collect_stats or not self.training:
            return
        dims = tuple(range(gate_pre.ndim - 1))
        gate_act = F.silu(gate_pre.float())
        up_metric = gate_act.square().mean(dim=dims)
        gate_metric = (up_pre.float() * silu_prime(gate_pre.float())).square().mean(dim=dims)
        down_metric = z.float().square().mean(dim=dims)
        _ema_(self.up_metric, up_metric, self.config.stats_beta, self.stats_seen)
        _ema_(self.gate_metric, gate_metric, self.config.stats_beta, self.stats_seen)
        _ema_(self.down_metric, down_metric, self.config.stats_beta, self.stats_seen)
        self.stats_seen.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_pre = self.gate_proj(x)
        up_pre = self.up_proj(x)
        z = F.silu(gate_pre) * up_pre
        self._update_stats(gate_pre.detach(), up_pre.detach(), z.detach())
        return self.down_proj(z)


class SparseMoE(nn.Module):
    """Simple top-1 research MoE used to test router/expert optimizer geometry."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.moe_experts <= 1:
            raise ValueError("SparseMoE requires at least two experts")
        self.config = config
        self.router = nn.Linear(config.n_embd, config.moe_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(config.moe_experts)])
        self.register_buffer("router_fisher", torch.ones(config.moe_experts), persistent=True)
        self.register_buffer(
            "expert_load", torch.full((config.moe_experts,), 1.0 / config.moe_experts), persistent=True
        )
        self.register_buffer("stats_seen", torch.zeros((), dtype=torch.long), persistent=True)
        self.collect_stats = False

    @torch.no_grad()
    def _update_router_stats(self, probs: torch.Tensor, top: torch.Tensor) -> None:
        if not self.collect_stats or not self.training:
            return
        fisher = categorical_fisher_diagonal(probs)
        load = torch.bincount(top.reshape(-1), minlength=self.config.moe_experts).float()
        load = load / load.sum().clamp_min(1.0)
        _ema_(self.router_fisher, fisher, self.config.stats_beta, self.stats_seen)
        _ema_(self.expert_load, load, self.config.stats_beta, self.stats_seen)
        self.stats_seen.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        probs = logits.softmax(dim=-1)
        weight, top = probs.max(dim=-1)
        self._update_router_stats(probs.detach(), top.detach())
        flat_x = x.reshape(-1, x.size(-1))
        flat_top = top.reshape(-1)
        flat_weight = weight.reshape(-1, 1)
        out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            mask = flat_top == expert_id
            if bool(mask.any()):
                expert_out = expert(flat_x[mask])
                out[mask] = expert_out * flat_weight[mask]
        return out.view_as(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = FunctionalAttention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SparseMoE(config) if config.moe_experts > 1 else SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class ResearchLM(nn.Module):
    """Compact modern decoder exposing an explicit ORBIT-all functional-role contract."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer(
            "token_frequency", torch.full((config.vocab_size,), 1.0 / config.vocab_size), persistent=True
        )
        self.register_buffer("token_stats_seen", torch.zeros((), dtype=torch.long), persistent=True)
        self.collect_stats = False
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith(("attn.o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def _update_token_frequency(self, input_ids: torch.Tensor, labels: torch.Tensor | None) -> None:
        if not self.collect_stats or not self.training:
            return
        tokens = input_ids.reshape(-1)
        if labels is not None:
            tokens = torch.cat((tokens, labels.reshape(-1)))
        counts = torch.bincount(tokens, minlength=self.config.vocab_size).float()
        counts = counts / counts.sum().clamp_min(1.0)
        _ema_(self.token_frequency, counts, self.config.stats_beta, self.token_stats_seen)
        self.token_stats_seen.add_(1)

    def set_stat_collection(self, enabled: bool) -> None:
        self.collect_stats = bool(enabled)
        for block in self.blocks:
            block.attn.collect_stats = bool(enabled)
            if isinstance(block.ffn, SparseMoE):
                block.ffn.collect_stats = bool(enabled)
                for expert in block.ffn.experts:
                    expert.collect_stats = bool(enabled)
            else:
                block.ffn.collect_stats = bool(enabled)

    @contextmanager
    def suspend_stat_collection(self):
        previous = self.collect_stats
        self.set_stat_collection(False)
        try:
            yield
        finally:
            self.set_stat_collection(previous)

    def functional_roles(self) -> dict[str, list[dict]]:
        roles: dict[str, list[dict]] = {
            "qk": [],
            "vo": [],
            "swiglu": [],
            "embedding": [],
            "rope": [],
            "moe": [],
        }
        roles["embedding"].append({"weight": self.token_embedding.weight, "module": self})
        for layer, block in enumerate(self.blocks):
            attn = block.attn
            roles["qk"].append(
                {"layer": layer, "q": attn.q_proj.weight, "k": attn.k_proj.weight, "module": attn}
            )
            roles["vo"].append(
                {"layer": layer, "v": attn.v_proj.weight, "o": attn.o_proj.weight, "module": attn}
            )
            if attn.log_inv_freq is not None:
                roles["rope"].append(
                    {"layer": layer, "frequency": attn.log_inv_freq, "module": attn}
                )
            if isinstance(block.ffn, SparseMoE):
                roles["moe"].append(
                    {"layer": layer, "router": block.ffn.router.weight, "module": block.ffn}
                )
                for expert_id, expert in enumerate(block.ffn.experts):
                    roles["swiglu"].append(
                        {
                            "layer": layer,
                            "expert": expert_id,
                            "gate": expert.gate_proj.weight,
                            "up": expert.up_proj.weight,
                            "down": expert.down_proj.weight,
                            "module": expert,
                        }
                    )
            else:
                ffn = block.ffn
                roles["swiglu"].append(
                    {
                        "layer": layer,
                        "gate": ffn.gate_proj.weight,
                        "up": ffn.up_proj.weight,
                        "down": ffn.down_proj.weight,
                        "module": ffn,
                    }
                )
        return roles

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        if input_ids.size(1) > self.config.block_size:
            raise ValueError("sequence length exceeds configured block_size")
        self._update_token_frequency(input_ids, labels)
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                def context_fn():
                    return nullcontext(), self.suspend_stat_collection()

                x = checkpoint(block, x, use_reentrant=False, context_fn=context_fn)
            else:
                x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)
