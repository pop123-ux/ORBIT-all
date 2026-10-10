from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_results(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text())
        row["_source"] = str(path)
        rows.append(row)
    return rows


def assert_compatible(rows: list[dict[str, Any]], phase: str) -> None:
    if not rows:
        raise RuntimeError(f"no results found for phase {phase!r}")
    digests = {row.get("protocol_digest") for row in rows}
    shas = {row.get("git_sha") for row in rows}
    if len(digests) != 1:
        raise RuntimeError(f"refusing to merge mixed protocol digests: {sorted(digests)}")
    if len(shas) != 1:
        raise RuntimeError(
            "refusing to merge mixed code revisions. Finish/discard the phase after a code change; "
            f"observed git SHAs: {sorted(shas)}"
        )
    if phase != "smoke" and any(bool(row.get("synthetic")) for row in rows):
        raise RuntimeError("synthetic rows cannot be merged into a non-smoke evidence phase")


def freeze_best(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_optimizer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_optimizer[row["optimizer"]].append(row)
    frozen: dict[str, dict[str, Any]] = {}
    for optimizer, opt_rows in sorted(by_optimizer.items()):
        candidate_ids = [r.get("candidate_id") for r in opt_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RuntimeError(
                f"duplicate matched-tune candidate for {optimizer}; do not average accidental reruns"
            )
        winner = min(opt_rows, key=lambda r: float(r["final_val_loss"]))
        frozen[optimizer] = {
            "candidate_id": winner.get("candidate_id"),
            "hparams": winner["hparams"],
            "tuning_seed": winner["seed"],
            "validation_loss": winner["final_val_loss"],
            "git_sha": winner["git_sha"],
            "protocol_digest": winner["protocol_digest"],
            "selection_rule": "minimum final validation loss over the shared predeclared candidate grid",
        }
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ORBIT-all campaign results with provenance gates")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    work = Path(args.work_dir)
    rows = load_results(work / "runs" / args.phase)
    assert_compatible(rows, args.phase)
    merged = work / "merged"
    merged.mkdir(parents=True, exist_ok=True)

    output = merged / f"{args.phase}.json"
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"merged {len(rows)} rows -> {output}")

    if args.phase == "matched_tune":
        frozen = freeze_best(rows)
        freeze_path = merged / "best_configs.json"
        if freeze_path.exists():
            existing = json.loads(freeze_path.read_text())
            if existing != frozen:
                raise RuntimeError(
                    "best_configs.json already exists with different contents. Never silently refreeze "
                    "recipes after downstream evaluation has started. Use a new work directory."
                )
        else:
            freeze_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
        print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
