from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbit_all.experiment import canonical_digest, git_sha, write_result_atomic
from orbit_all.production_benchmark import load_production_protocol, run_production_training


CORE_PATHS = (
    "src/orbit_all/geometry.py",
    "src/orbit_all/model.py",
    "src/orbit_all/optimizer.py",
    "src/orbit_all/experiment.py",
    "configs/protocol.json",
)


@dataclass(frozen=True)
class Task:
    phase: str
    optimizer: str
    hparams: dict[str, Any]
    seed: int
    steps: int

    @property
    def task_id(self) -> str:
        candidate = str(self.hparams["id"])
        return f"production_{self.phase}__{self.optimizer}__{candidate}__s{self.seed}"


def assert_frozen_core(base_sha: str) -> None:
    command = ["git", "diff", "--quiet", base_sha, "HEAD", "--", *CORE_PATHS]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "production benchmark branch modifies frozen ORBIT-all core files. "
            "Keep this extension additive or start a separately versioned campaign."
        )


def shard_owner(task_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(task_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def _all_optimizers(protocol: dict[str, Any]) -> list[str]:
    return list(protocol["optimizers"])


def _load_best(work_dir: Path, protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = work_dir / "merged" / "production_best_configs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Finish production tune and merge it before confirmation."
        )
    best = json.loads(path.read_text())
    expected = set(_all_optimizers(protocol))
    if set(best) != expected:
        raise RuntimeError(f"frozen production recipes mismatch: expected={sorted(expected)}")
    current_sha = git_sha()
    protocol_digest = canonical_digest(protocol)
    for name, entry in best.items():
        if entry.get("git_sha") != current_sha:
            raise RuntimeError(
                f"{name} was tuned under git_sha={entry.get('git_sha')}, current={current_sha}. "
                "Do not carry benchmark recipes across code revisions."
            )
        if entry.get("protocol_digest") != protocol_digest:
            raise RuntimeError(f"{name} protocol digest differs from current benchmark protocol")
    return best


def build_tasks(phase: str, protocol: dict[str, Any], work_dir: Path) -> list[Task]:
    if phase == "smoke":
        return [
            Task(
                phase="smoke",
                optimizer=name,
                hparams=dict(protocol["grids"][name][0]),
                seed=0,
                steps=int(protocol["phases"]["smoke"]["steps"]),
            )
            for name in _all_optimizers(protocol)
        ]
    if phase == "tune":
        spec = protocol["phases"]["tune"]
        return [
            Task(
                phase="tune",
                optimizer=name,
                hparams=dict(candidate),
                seed=int(spec["seed"]),
                steps=int(spec["steps"]),
            )
            for name in _all_optimizers(protocol)
            for candidate in protocol["grids"][name]
        ]
    if phase == "confirm":
        spec = protocol["phases"]["confirm"]
        best = _load_best(work_dir, protocol)
        return [
            Task(
                phase="confirm",
                optimizer=name,
                hparams=dict(best[name]["hparams"]),
                seed=int(seed),
                steps=int(spec["steps"]),
            )
            for name in _all_optimizers(protocol)
            for seed in spec["seeds"]
        ]
    raise ValueError(f"unknown phase {phase!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-confirmation production optimizer benchmark")
    parser.add_argument("--phase", required=True, choices=("smoke", "tune", "confirm"))
    parser.add_argument("--work-dir", required=True, help="new benchmark result directory")
    parser.add_argument(
        "--data-work-dir",
        required=True,
        help="existing frozen ORBIT-all work directory containing data/train.bin and data/val.bin",
    )
    parser.add_argument("--protocol", default="configs/production_optimizer_protocol.json")
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

    protocol = load_production_protocol(args.protocol)
    assert_frozen_core(str(protocol["core_base_sha"]))
    protocol_digest = canonical_digest(protocol)
    current_sha = git_sha()
    work_dir = Path(args.work_dir)
    tasks = build_tasks(args.phase, protocol, work_dir)
    owned = [task for task in tasks if shard_owner(task.task_id, args.num_shards) == args.shard_id]
    if args.max_tasks is not None:
        owned = owned[: args.max_tasks]

    print(
        f"phase={args.phase} total_tasks={len(tasks)} owned={len(owned)} "
        f"shard={args.shard_id}/{args.num_shards}"
    )
    for task in owned:
        print(task.task_id)
    if args.list_tasks:
        return

    synthetic = bool(args.synthetic or args.phase == "smoke")
    if args.phase != "smoke" and synthetic:
        print("WARNING: synthetic benchmark rows are plumbing-only and not paper evidence")

    for index, task in enumerate(owned, start=1):
        out_path = work_dir / "runs" / task.phase / f"{task.task_id}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing.get("git_sha") != current_sha:
                raise RuntimeError(f"stale result git SHA for {task.task_id}")
            if existing.get("protocol_digest") != protocol_digest:
                raise RuntimeError(f"stale result protocol digest for {task.task_id}")
            print(f"[{index}/{len(owned)}] skip complete {task.task_id}")
            continue

        print(f"[{index}/{len(owned)}] run {task.task_id}", flush=True)
        result = run_production_training(
            protocol=protocol,
            optimizer_name=task.optimizer,
            hparams=task.hparams,
            seed=task.seed,
            steps=task.steps,
            data_work_dir=args.data_work_dir,
            device_name=args.device,
            synthetic=synthetic,
            steps_override=args.steps_override,
        )
        result["phase"] = task.phase
        result["task_id"] = task.task_id
        result["candidate_id"] = task.hparams["id"]
        write_result_atomic(out_path, result)
        print(
            f"  val={result['final_val_loss']:.6f} wall={result['wall_seconds']:.1f}s "
            f"peak={result['peak_cuda_gib']:.3f}GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
