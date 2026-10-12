from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from orbit_all.experiment import canonical_digest, load_protocol, run_training, write_result_atomic


CORE_PATHS = (
    "src/orbit_all/geometry.py",
    "src/orbit_all/model.py",
    "src/orbit_all/optimizer.py",
    "src/orbit_all/experiment.py",
    "configs/protocol.json",
)


def shard_owner(task_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(task_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def assert_core_matches_base(base_sha: str) -> None:
    proc = subprocess.run(
        ["git", "diff", "--quiet", base_sha, "HEAD", "--", *CORE_PATHS],
        check=False,
    )
    if proc.returncode == 0:
        return
    if proc.returncode == 1:
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", base_sha, "HEAD", "--", *CORE_PATHS],
            text=True,
        ).strip()
        raise RuntimeError(
            "legacy ORBIT comparison requires the frozen experiment core to match the original "
            f"ORBIT-all confirmation revision {base_sha}. Changed core files:\n{changed}"
        )
    raise RuntimeError("git diff failed while checking the frozen experiment core")


def validate_existing_campaign(work: Path, base_sha: str, protocol: dict[str, Any]) -> None:
    best_path = work / "merged" / "best_configs.json"
    confirm_path = work / "merged" / "confirm.json"
    manifest_path = work / "data" / "manifest.json"
    for path in (best_path, confirm_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"required frozen campaign artifact is missing: {path}")

    frozen = json.loads(best_path.read_text())
    tuned_shas = {entry.get("git_sha") for entry in frozen.values()}
    if tuned_shas != {base_sha}:
        raise RuntimeError(
            f"best_configs.json was not frozen at expected base SHA {base_sha}: {sorted(tuned_shas)}"
        )

    confirm = json.loads(confirm_path.read_text())
    confirm_shas = {row.get("git_sha") for row in confirm}
    if confirm_shas != {base_sha}:
        raise RuntimeError(
            f"confirm.json does not come entirely from expected base SHA {base_sha}: "
            f"{sorted(confirm_shas)}"
        )

    expected_digest = canonical_digest(protocol)
    confirm_digests = {row.get("protocol_digest") for row in confirm}
    if confirm_digests != {expected_digest}:
        raise RuntimeError("confirmation protocol digest differs from the currently loaded frozen protocol")

    manifest = json.loads(manifest_path.read_text())
    expected_revision = protocol["data"]["revision"]
    if manifest.get("revision") != expected_revision:
        raise RuntimeError(
            "dataset manifest revision differs from the frozen protocol: "
            f"manifest={manifest.get('revision')}, protocol={expected_revision}"
        )
    if not manifest.get("paper_mode"):
        raise RuntimeError("dataset manifest is not marked paper_mode=true")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the published ORBIT paper recipe inside the frozen ORBIT-all harness"
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--protocol", default="configs/protocol.json")
    parser.add_argument("--baseline-config", default="configs/legacy_orbit_paper_recipe.json")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("shard-id must satisfy 0 <= shard-id < num-shards")

    protocol = load_protocol(args.protocol)
    baseline = json.loads(Path(args.baseline_config).read_text())
    work = Path(args.work_dir)
    base_sha = str(baseline["experiment_base_sha"])
    assert_core_matches_base(base_sha)
    validate_existing_campaign(work, base_sha, protocol)

    model_name = str(baseline["model"])
    steps = int(baseline["steps"])
    implementation_mode = str(baseline["implementation_mode"])
    result_label = str(baseline["result_label"])
    hparams = dict(baseline["hparams"])
    seeds = [int(seed) for seed in baseline["seeds"]]
    current_sha = git_sha()
    protocol_digest = canonical_digest(protocol)

    tasks = []
    for seed in seeds:
        task_id = f"legacy_orbit_compare__{model_name}__{result_label}__{hparams['id']}__s{seed}"
        if shard_owner(task_id, args.num_shards) == args.shard_id:
            tasks.append((task_id, seed))

    print(
        f"phase=legacy_orbit_compare total_tasks={len(seeds)} owned={len(tasks)} "
        f"shard={args.shard_id}/{args.num_shards}"
    )
    for task_id, _ in tasks:
        print(task_id)
    if args.list_tasks:
        return

    out_dir = work / "runs" / "legacy_orbit_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, (task_id, seed) in enumerate(tasks, start=1):
        out_path = out_dir / f"{task_id}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if (
                existing.get("git_sha") != current_sha
                or existing.get("protocol_digest") != protocol_digest
                or existing.get("experiment_base_sha") != base_sha
            ):
                raise RuntimeError(
                    f"stale legacy baseline result exists for {task_id}; use a new work directory"
                )
            print(f"[{index}/{len(tasks)}] skip complete {task_id}")
            continue

        print(f"[{index}/{len(tasks)}] run {task_id}", flush=True)
        result = run_training(
            protocol=protocol,
            model_name=model_name,
            optimizer_mode=implementation_mode,
            hparams=hparams,
            seed=seed,
            steps=steps,
            work_dir=work,
            device_name=args.device,
            synthetic=False,
        )
        result["phase"] = "legacy_orbit_compare"
        result["task_id"] = task_id
        result["candidate_id"] = hparams["id"]
        result["implementation_mode"] = implementation_mode
        result["optimizer"] = result_label
        result["experiment_base_sha"] = base_sha
        result["source_orbit_repository"] = baseline["source"]["repository"]
        result["source_orbit_commit"] = baseline["source"]["commit"]
        result["source_orbit_config_path"] = baseline["source"]["config_path"]
        result["baseline_interpretation"] = (
            "Published ORBIT Q/K recipe transferred to the frozen ORBIT-all dense harness; "
            "not an exact historical paper-stack reproduction."
        )
        write_result_atomic(out_path, result)
        print(
            f"  val={result['final_val_loss']:.6f} wall={result['wall_seconds']:.1f}s "
            f"peak={result['peak_cuda_gib']:.3f}GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
