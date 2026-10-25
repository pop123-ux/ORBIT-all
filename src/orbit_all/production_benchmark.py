from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .experiment import (
    canonical_digest,
    evaluate,
    git_sha,
    load_token_file,
    lr_multiplier,
    sample_batch,
    seed_everything,
)
from .geometry import count_parameters
from .model import ModelConfig, ResearchLM
from .production_optimizers import (
    build_production_optimizer,
    optimizer_base_lr,
    optimizer_diagnostics,
    set_optimizer_lr,
)


def load_production_protocol(
    path: str | Path = "configs/production_optimizer_protocol.json",
) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _model_config(protocol: dict[str, Any], *, cpu_smoke: bool) -> ModelConfig:
    raw = dict(protocol["model"])
    raw.pop("batch_size", None)
    raw.pop("grad_accum", None)
    config = ModelConfig(**raw)
    if not cpu_smoke:
        return config
    return ModelConfig(
        vocab_size=min(config.vocab_size, 1024),
        block_size=min(config.block_size, 64),
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        mlp_hidden=256,
        dropout=0.0,
        bias=config.bias,
        rope_base=config.rope_base,
        rope_mode=config.rope_mode,
        qk_norm=config.qk_norm,
        stats_beta=config.stats_beta,
        moe_experts=0,
        gradient_checkpointing=False,
    )


def run_production_training(
    *,
    protocol: dict[str, Any],
    optimizer_name: str,
    hparams: dict[str, Any],
    seed: int,
    steps: int,
    data_work_dir: str | Path,
    device_name: str = "cuda",
    synthetic: bool = False,
    steps_override: int | None = None,
    eval_batches: int = 12,
) -> dict[str, Any]:
    seed_everything(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for smoke tests")
    device = torch.device(device_name)
    cpu_smoke = device.type == "cpu" and steps_override is not None
    config = _model_config(protocol, cpu_smoke=cpu_smoke)
    batch_size = 2 if cpu_smoke else int(protocol["model"].get("batch_size", 4))
    grad_accum = 1 if cpu_smoke else int(protocol["model"].get("grad_accum", 1))

    model = ResearchLM(config).to(device)
    optimizer = build_production_optimizer(
        model,
        optimizer_name,
        hparams,
        protocol["optimizer_shared"],
    )
    base_lr = optimizer_base_lr(optimizer_name, hparams)
    total_steps = int(steps_override if steps_override is not None else steps)

    train_tokens: np.ndarray | None = None
    val_tokens: np.ndarray | None = None
    data_work_dir = Path(data_work_dir)
    if not synthetic:
        train_tokens = load_token_file(data_work_dir / "data" / "train.bin")
        val_tokens = load_token_file(data_work_dir / "data" / "val.bin")

    train_gen = torch.Generator(device="cpu").manual_seed(seed * 10_003 + 17)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    recent_losses: list[float] = []

    for step in range(total_steps):
        multiplier = lr_multiplier(
            step,
            total_steps,
            warmup_ratio=float(protocol["schedule"]["warmup_ratio"]),
        )
        set_optimizer_lr(optimizer, optimizer_name, base_lr, multiplier)
        accumulated = 0.0
        for _ in range(grad_accum):
            batch = sample_batch(
                train_tokens,
                batch_size=batch_size,
                seq_len=config.block_size,
                vocab_size=config.vocab_size,
                generator=train_gen,
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                out = model(batch, batch)
                loss = out.loss / grad_accum
            scaler.scale(loss).backward()
            accumulated += float(loss.detach().float().cpu())
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        recent_losses.append(accumulated)
        if len(recent_losses) > 50:
            recent_losses.pop(0)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - start
    final_val = evaluate(
        model,
        val_tokens,
        batches=eval_batches if not synthetic else 2,
        batch_size=batch_size,
        seq_len=config.block_size,
        seed=seed * 20_011 + 29,
        device=device,
    )
    peak_gib = (
        float(torch.cuda.max_memory_allocated(device)) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    provenance = dict(protocol["optimizers"][optimizer_name]["provenance"])
    return {
        "schema": 1,
        "project": protocol["name"],
        "benchmark_kind": "post-confirmation production-optimizer extension",
        "core_base_sha": protocol["core_base_sha"],
        "git_sha": git_sha(),
        "protocol_digest": canonical_digest(protocol),
        "optimizer": optimizer_name,
        "optimizer_provenance": provenance,
        "hparams": dict(hparams),
        "seed": int(seed),
        "steps": total_steps,
        "model_config": asdict(config),
        "parameter_count": count_parameters(model),
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "tokens_per_update": batch_size * grad_accum * config.block_size,
        "synthetic": bool(synthetic),
        "final_train_loss_50": float(np.mean(recent_losses)),
        "final_val_loss": final_val,
        "wall_seconds": wall_seconds,
        "peak_cuda_gib": peak_gib,
        "optimizer_diagnostics": optimizer_diagnostics(optimizer),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
