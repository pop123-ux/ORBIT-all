from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def paired_summary(rows: list[dict[str, Any]], candidate: str, reference: str) -> dict[str, Any]:
    by_key: dict[tuple[int, int | None], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["seed"]), row.get("block_size_override"))
        by_key.setdefault(key, {})[row["optimizer"]] = row

    pairs = []
    for key in sorted(by_key):
        cell = by_key[key]
        if candidate in cell and reference in cell:
            diff = float(cell[candidate]["final_val_loss"]) - float(cell[reference]["final_val_loss"])
            pairs.append(
                {
                    "seed": key[0],
                    "block_size": key[1],
                    "candidate_loss": float(cell[candidate]["final_val_loss"]),
                    "reference_loss": float(cell[reference]["final_val_loss"]),
                    "difference": diff,
                }
            )
    if len(pairs) < 2:
        raise RuntimeError(f"need at least two paired rows for {candidate} vs {reference}")
    d = np.asarray([p["difference"] for p in pairs], dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    ci = [mean - tcrit * se, mean + tcrit * se]
    t_stat, p_value = stats.ttest_rel(
        [p["candidate_loss"] for p in pairs],
        [p["reference_loss"] for p in pairs],
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "definition": "difference = candidate final validation loss minus reference final validation loss",
        "n_pairs": n,
        "mean_difference": mean,
        "sample_sd_difference": sd,
        "standard_error": se,
        "student_t_df": n - 1,
        "student_t_critical_95": tcrit,
        "ci_95": ci,
        "wins_candidate_lower": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
        "two_sided_paired_t_pvalue": float(p_value),
        "paired_t_statistic": float(t_stat),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired ORBIT-all statistical analysis")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--phase", default="confirm")
    parser.add_argument("--candidate", default="orbit_all")
    parser.add_argument("--reference", default="muon")
    args = parser.parse_args()
    work = Path(args.work_dir)
    rows = load_rows(work / "merged" / f"{args.phase}.json")
    summary = paired_summary(rows, args.candidate, args.reference)
    out_dir = work / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.phase}__{args.candidate}_vs_{args.reference}.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
