from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from orbit_all.experiment import canonical_digest, git_sha
from orbit_all.production_benchmark import load_production_protocol


def _expected_count(protocol: dict[str, Any], phase: str) -> int:
    if phase == "tune":
        return sum(len(protocol["grids"][name]) for name in protocol["optimizers"])
    if phase == "confirm":
        return len(protocol["optimizers"]) * len(protocol["phases"]["confirm"]["seeds"])
    raise ValueError(phase)


def load_rows(work_dir: Path, phase: str) -> list[dict[str, Any]]:
    root = work_dir / "runs" / phase
    if not root.exists():
        raise FileNotFoundError(root)
    return [json.loads(path.read_text()) for path in sorted(root.glob("*.json"))]


def validate_rows(rows: list[dict[str, Any]], protocol: dict[str, Any], phase: str) -> None:
    expected_count = _expected_count(protocol, phase)
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} {phase} rows, found {len(rows)}")
    if any(bool(row.get("synthetic")) for row in rows):
        raise RuntimeError("synthetic rows cannot be merged as production benchmark evidence")
    shas = {row.get("git_sha") for row in rows}
    if shas != {git_sha()}:
        raise RuntimeError(f"mixed/stale git SHAs: {sorted(str(x) for x in shas)}")
    expected_digest = canonical_digest(protocol)
    if {row.get("protocol_digest") for row in rows} != {expected_digest}:
        raise RuntimeError("mixed/stale production benchmark protocol digests")
    task_ids = [row.get("task_id") for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("duplicate task IDs found")
    expected_optimizers = set(protocol["optimizers"])
    if {row.get("optimizer") for row in rows} != expected_optimizers:
        raise RuntimeError("optimizer coverage is incomplete")


def select_best(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for optimizer in protocol["optimizers"]:
        candidates = [row for row in rows if row["optimizer"] == optimizer]
        winner = min(candidates, key=lambda row: float(row["final_val_loss"]))
        best[optimizer] = {
            "candidate_id": winner["candidate_id"],
            "hparams": winner["hparams"],
            "final_val_loss": winner["final_val_loss"],
            "git_sha": winner["git_sha"],
            "protocol_digest": winner["protocol_digest"],
            "selection": "minimum final validation loss on the predeclared tuning seed",
        }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge production optimizer benchmark rows")
    parser.add_argument("--phase", required=True, choices=("tune", "confirm"))
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--protocol", default="configs/production_optimizer_protocol.json")
    args = parser.parse_args()

    protocol = load_production_protocol(args.protocol)
    work_dir = Path(args.work_dir)
    rows = load_rows(work_dir, args.phase)
    validate_rows(rows, protocol, args.phase)

    merged_dir = work_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    out = merged_dir / f"production_{args.phase}.json"
    out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"merged {len(rows)} rows -> {out}")

    if args.phase == "tune":
        best = select_best(rows, protocol)
        best_path = merged_dir / "production_best_configs.json"
        if best_path.exists():
            existing = json.loads(best_path.read_text())
            if existing != best:
                raise RuntimeError(
                    "production_best_configs.json already exists with different winners; "
                    "do not silently change frozen recipes"
                )
        else:
            best_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
        print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
