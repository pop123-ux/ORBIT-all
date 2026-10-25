from __future__ import annotations

import torch

from orbit_all.model import ModelConfig, ResearchLM
from orbit_all.production_optimizers import (
    MoonshotMuon,
    PaLMAdafactor,
    build_production_optimizer,
)


def tiny_model() -> ResearchLM:
    return ResearchLM(
        ModelConfig(
            vocab_size=128,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            n_embd=16,
            mlp_hidden=32,
            dropout=0.0,
            bias=False,
        )
    )


def shared() -> dict:
    return {
        "momentum": 0.95,
        "ns_steps": 2,
        "functional_power": 1.0,
        "metric_eps": 1e-5,
        "metric_condition_cap": 100.0,
        "moonshot_rms_scale": 0.2,
        "qk_clip_threshold": 100.0,
    }


def run_one_step(model: ResearchLM, optimizer) -> None:
    tokens = torch.randint(0, model.config.vocab_size, (2, model.config.block_size))
    loss = model(tokens, tokens).loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert all(torch.isfinite(param).all() for param in model.parameters())


def test_all_production_optimizer_builders_step() -> None:
    configs = {
        "orbit_all": {"matrix_lr": 0.01, "aux_lr": 0.001, "weight_decay": 0.01},
        "muon_moonlight": {
            "matrix_lr": 0.01,
            "aux_lr": 0.001,
            "weight_decay": 0.01,
        },
        "muonclip_kimi": {
            "matrix_lr": 0.01,
            "aux_lr": 0.001,
            "weight_decay": 0.01,
        },
        "adamw": {"lr": 1e-3, "weight_decay": 0.01},
        "lion": {"lr": 1e-4, "weight_decay": 0.01},
        "adafactor_palm": {"lr": 1e-2, "weight_decay": 0.01},
        "adam": {"lr": 1e-3, "weight_decay": 0.01},
    }
    for name, hparams in configs.items():
        model = tiny_model()
        optimizer = build_production_optimizer(model, name, hparams, shared())
        run_one_step(model, optimizer)


def test_kimi_qk_clip_records_and_clips() -> None:
    model = tiny_model()
    optimizer = MoonshotMuon(
        model,
        matrix_lr=0.01,
        aux_lr=0.001,
        weight_decay=0.01,
        ns_steps=2,
        qk_clip=True,
        qk_clip_threshold=1e-6,
    )
    run_one_step(model, optimizer)
    diagnostics = optimizer.diagnostics()
    assert diagnostics["qk_clip_fraction"] > 0.0
    assert diagnostics["qk_max_logit"] > 0.0


def test_palm_adafactor_uses_dynamic_second_moment() -> None:
    param = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = PaLMAdafactor([param], lr=0.01)
    for _ in range(2):
        param.grad = torch.full_like(param, 0.5)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert optimizer.state[param]["step"] == 2
    assert torch.isfinite(param).all()
