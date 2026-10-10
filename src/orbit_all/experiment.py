from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .geometry import count_parameters
from .model import ModelConfig, ResearchLM
from .optimizer import OrbitAll


def load_protocol(path: str | Path = "configs/protocol.json") -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config_from_protocol(protocol: dict[str, Any], name: str, **overrides: Any) -> ModelConfig:
    raw = dict(protocol["model"][name])
    for key in ("batch_size", "grad_accum"):
        raw.pop(key, None)
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return ModelConfig(**raw)


def sample_batch(
    tokens: np.ndarray | None,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if tokens is None:
        # Keep RNG on CPU so the same synthetic batch stream is portable across devices.
        batch = torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len),
            generator=generator,
            device="cpu",
        )
        return batch.to(device)
    if len(tokens) <= seq_len + 1:
        raise ValueError("token file is too short for requested sequence length")
    starts = torch.randint(
        0,
        len(tokens) - seq_len - 1,
        (batch_size,),
        generator=generator,
        device="cpu",
    ).tolist()
    batch = np.stack([tokens[i : i + seq_len] for i in starts])
    return torch.from_numpy(batch.astype(np.int64, copy=False)).to(device)


def load_token_file(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Run scripts/prepare_data.py first or pass --synthetic for plumbing tests."
        )
    return np.memmap(path, dtype=np.int32, mode="r")


def lr_multiplier(step: int, total_steps: int, warmup_ratio: float = 0.05) -> float:
    warmup = max(1, int(total_steps * warmup_ratio))
    if step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model: ResearchLM,
    tokens: np.ndarray | None,
    *,
    batches: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    amp = device.type == "cuda"
    for _ in range(batches):
        batch = sample_batch(
            tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=model.config.vocab_size,
            generator=generator,
            device=device,
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(batch, batch)
        losses.append(float(out.loss.detach().float().cpu()))
    model.train(was_training)
    return float(np.mean(losses))


def _context_batch_policy(model_entry: dict[str, Any], block_size: int) -> tuple[int, int]:
    base_batch = int(model_entry.get("batch_size", 4))
    base_accum = int(model_entry.get("grad_accum", 1))
    base_block = int(model_entry["block_size"])
    target_tokens = base_batch * base_accum * base_block
    batch = max(1, (base_batch * base_block) // block_size)
    accum = max(1, math.ceil(target_tokens / max(1, batch * block_size)))
    return batch, accum


def run_training(
    *,
    protocol: dict[str, Any],
    model_name: str,
    optimizer_mode: str,
    hparams: dict[str, Any],
    seed: int,
    steps: int,
    work_dir: str | Path,
    device_name: str = "cuda",
    synthetic: bool = False,
    steps_override: int | None = None,
    block_size_override: int | None = None,
    eval_batches: int = 12,
) -> dict[str, Any]:
    seed_everything(seed)
    work_dir = Path(work_dir)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available; use --device cpu for smoke tests")
    device = torch.device(device_name)
    model_entry = protocol["model"][model_name]
    block_size = int(block_size_override or model_entry["block_size"])
    if block_size_override and block_size_override > int(model_entry["block_size"]):
        batch_size, grad_accum = _context_batch_policy(model_entry, block_size)
        use_checkpointing = True
    else:
        batch_size = int(model_entry.get("batch_size", 4))
        grad_accum = int(model_entry.get("grad_accum", 1))
        use_checkpointing = bool(model_entry.get("gradient_checkpointing", False))

    config = model_config_from_protocol(
        protocol,
        model_name,
        block_size=block_size,
        gradient_checkpointing=use_checkpointing,
    )
    if device.type == "cpu" and steps_override is not None:
        # CPU smoke validates plumbing without instantiating the 124M-class network.
        config = ModelConfig(
            vocab_size=min(config.vocab_size, 1024),
            block_size=min(config.block_size, 64),
            n_layer=2,
            n_head=4,
            n_kv_head=2 if config.n_kv_head < config.n_head else 4,
            n_embd=128,
            mlp_hidden=256,
            rope_mode=config.rope_mode,
            qk_norm=config.qk_norm,
            stats_beta=config.stats_beta,
            moe_experts=config.moe_experts,
            gradient_checkpointing=False,
        )
        batch_size, grad_accum = 2, 1

    model = ResearchLM(config).to(device)
    optimizer = OrbitAll(
        model,
        mode=optimizer_mode,
        matrix_lr=float(hparams["matrix_lr"]),
        aux_lr=float(hparams["aux_lr"]),
        weight_decay=float(hparams["weight_decay"]),
        momentum=float(protocol["optimizer"]["shared"]["momentum"]),
        ns_steps=int(protocol["optimizer"]["shared"]["ns_steps"]),
        functional_power=float(protocol["optimizer"]["shared"]["functional_power"]),
        metric_eps=float(protocol["optimizer"]["shared"]["metric_eps"]),
        metric_condition_cap=float(protocol["optimizer"]["shared"]["metric_condition_cap"]),
    )
    base_lr = float(hparams["matrix_lr"])
    total_steps = int(steps_override if steps_override is not None else steps)

    train_tokens: np.ndarray | None = None
    val_tokens: np.ndarray | None = None
    if not synthetic:
        train_tokens = load_token_file(work_dir / "data" / "train.bin")
        val_tokens = load_token_file(work_dir / "data" / "val.bin")

    train_gen = torch.Generator(device="cpu").manual_seed(seed * 10_003 + 17)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    recent_losses: list[float] = []

    for step in range(total_steps):
        multiplier = lr_multiplier(step, total_steps)
        optimizer.param_groups[0]["lr"] = base_lr * multiplier
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
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
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
        float(torch.cuda.max_memory_allocated(device)) / (1024**3) if device.type == "cuda" else 0.0
    )

    return {
        "schema": 1,
        "project": protocol["name"],
        "git_sha": git_sha(),
        "protocol_digest": canonical_digest(protocol),
        "model_name": model_name,
        "model_config": asdict(config),
        "parameter_count": count_parameters(model),
        "optimizer": optimizer_mode,
        "families": sorted(optimizer.families),
        "hparams": dict(hparams),
        "seed": int(seed),
        "steps": total_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "tokens_per_update": batch_size * grad_accum * config.block_size,
        "synthetic": bool(synthetic),
        "final_train_loss_50": float(np.mean(recent_losses)),
        "final_val_loss": final_val,
        "wall_seconds": wall_seconds,
        "peak_cuda_gib": peak_gib,
        "optimizer_diagnostics": optimizer.diagnostics(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def write_result_atomic(path: str | Path, result: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)
