from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbit_all.experiment import (
    canonical_digest,
    git_sha,
    load_protocol,
    run_training,
    write_result_atomic,
)


@dataclass(frozen=True)
class Task:
    phase: str
    model_name: str
    optimizer: str
    hparams: dict[str, Any]
    seed: int
    steps: int
    block_size: int | None = None

    @property
    def task_id(self) -> str:
        candidate = str(self.hparams.get("id", "frozen"))
        context = f"-ctx{self.block_size}" if self.block_size else ""
        return f"{self.phase}__{self.model_name}__{self.optimizer}__{candidate}__s{self.seed}{context}"


def shard_owner(task_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(task_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def load_frozen(work_dir: Path) -> dict[str, dict[str, Any]]:
    path = work_dir / "merged" / "best_configs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Finish matched_tune and run scripts/merge.py --phase matched_tune first."
        )
    frozen = json.loads(path.read_text())
    current = git_sha()
    tuning_shas = {entry.get("git_sha") for entry in frozen.values()}
    if current != "unknown" and tuning_shas != {current}:
        raise RuntimeError(
            "current code revision differs from the frozen matched-tuning revision. "
            "Start a new campaign/work directory and retune rather than carrying recipes across code changes. "
            f"current={current}, tuned={sorted(tuning_shas)}"
        )
    return frozen


def grid(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(x) for x in protocol["matched_search_grid"]]


def frozen_hparams(frozen: dict[str, dict[str, Any]], optimizer: str) -> dict[str, Any]:
    if optimizer not in frozen:
        raise KeyError(f"no frozen recipe for {optimizer!r}")
    return dict(frozen[optimizer]["hparams"])


def build_tasks(phase: str, protocol: dict[str, Any], work_dir: Path) -> list[Task]:
    phases = protocol["phases"]
    if phase == "smoke":
        return [Task("smoke", "dense_124m", "orbit_all", dict(grid(protocol)[4]), 0, 2)]

    if phase == "discovery":
        spec = phases[phase]
        return [
            Task(phase, spec["model"], optimizer, dict(candidate), int(spec["seed"]), int(spec["steps"]))
            for optimizer in spec["optimizers"]
            for candidate in grid(protocol)[::2]
        ]

    if phase == "matched_tune":
        spec = phases[phase]
        return [
            Task(phase, spec["model"], optimizer, dict(candidate), int(spec["seed"]), int(spec["steps"]))
            for optimizer in spec["optimizers"]
            for candidate in grid(protocol)
        ]

    frozen = load_frozen(work_dir)
    if phase == "confirm":
        spec = phases[phase]
        return [
            Task(
                phase,
                spec["model"],
                optimizer,
                frozen_hparams(frozen, optimizer),
                int(seed),
                int(spec["steps"]),
            )
            for optimizer in ("muon", "orbit_qk", "orbit_all")
            for seed in spec["seeds"]
        ]

    if phase == "family_ablation":
        spec = phases[phase]
        hparams = frozen_hparams(frozen, "orbit_all")
        return [
            Task(phase, spec["model"], variant, dict(hparams), int(seed), int(spec["steps"]))
            for variant in spec["variants"]
            for seed in spec["seeds"]
        ]

    transfer_phases = {"gqa_transfer", "qknorm_transfer", "rope_transfer", "moe_transfer"}
    if phase in transfer_phases:
        spec = phases[phase]
        return [
            Task(
                phase,
                spec["model"],
                optimizer,
                frozen_hparams(frozen, optimizer),
                int(seed),
                int(spec["steps"]),
            )
            for optimizer in ("muon", "orbit_qk", "orbit_all")
            for seed in spec["seeds"]
        ]

    if phase == "context_transfer":
        spec = phases[phase]
        return [
            Task(
                phase,
                spec["model"],
                optimizer,
                frozen_hparams(frozen, optimizer),
                int(seed),
                int(spec["steps"]),
                int(block_size),
            )
            for block_size in spec["block_sizes"]
            for optimizer in ("muon", "orbit_qk", "orbit_all")
            for seed in spec["seeds"]
        ]

    if phase == "horizon":
        spec = phases[phase]
        return [
            Task(
                phase,
                spec["model"],
                optimizer,
                frozen_hparams(frozen, optimizer),
                int(seed),
                int(spec["steps"]),
            )
            for optimizer in spec["optimizers"]
            for seed in spec["seeds"]
        ]

    raise ValueError(f"unknown phase {phase!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ORBIT-all staged paper campaign")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--protocol", default="configs/protocol.json")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--steps-override", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("shard-id must satisfy 0 <= shard-id < num-shards")
    protocol = load_protocol(args.protocol)
    protocol_digest = canonical_digest(protocol)
    current_sha = git_sha()
    work_dir = Path(args.work_dir)
    tasks = build_tasks(args.phase, protocol, work_dir)
    owned = [t for t in tasks if shard_owner(t.task_id, args.num_shards) == args.shard_id]
    if args.max_tasks is not None:
        owned = owned[: args.max_tasks]

    print(f"phase={args.phase} total_tasks={len(tasks)} owned={len(owned)} shard={args.shard_id}/{args.num_shards}")
    for task in owned:
        print(task.task_id)
    if args.list_tasks:
        return

    if args.phase != "smoke" and args.synthetic:
        print("WARNING: synthetic results are plumbing diagnostics and must not be used as paper evidence")
    synthetic = bool(args.synthetic or args.phase == "smoke")
    for index, task in enumerate(owned, start=1):
        out_path = work_dir / "runs" / task.phase / f"{task.task_id}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing.get("git_sha") != current_sha or existing.get("protocol_digest") != protocol_digest:
                raise RuntimeError(
                    f"stale result exists for {task.task_id}. Do not overwrite/reuse a task across code or "
                    "protocol changes; choose a new work directory."
                )
            print(f"[{index}/{len(owned)}] skip complete {task.task_id}")
            continue
        print(f"[{index}/{len(owned)}] run {task.task_id}", flush=True)
        result = run_training(
            protocol=protocol,
            model_name=task.model_name,
            optimizer_mode=task.optimizer,
            hparams=task.hparams,
            seed=task.seed,
            steps=task.steps,
            work_dir=work_dir,
            device_name=args.device,
            synthetic=synthetic,
            steps_override=args.steps_override,
            block_size_override=task.block_size,
        )
        result["phase"] = task.phase
        result["task_id"] = task.task_id
        result["candidate_id"] = task.hparams.get("id")
        result["block_size_override"] = task.block_size
        write_result_atomic(out_path, result)
        print(
            f"  val={result['final_val_loss']:.6f} wall={result['wall_seconds']:.1f}s "
            f"peak={result['peak_cuda_gib']:.3f}GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
