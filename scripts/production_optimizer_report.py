from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from orbit_all.production_benchmark import load_production_protocol


def paired_summary(
    rows: list[dict[str, Any]], candidate: str, reference: str
) -> dict[str, Any]:
    candidate_rows = {
        int(row["seed"]): row for row in rows if row["optimizer"] == candidate
    }
    reference_rows = {
        int(row["seed"]): row for row in rows if row["optimizer"] == reference
    }
    seeds = sorted(set(candidate_rows) & set(reference_rows))
    if len(seeds) < 2:
        raise RuntimeError(f"need at least two paired rows for {candidate} vs {reference}")
    pairs = []
    differences = []
    for seed in seeds:
        cand = float(candidate_rows[seed]["final_val_loss"])
        ref = float(reference_rows[seed]["final_val_loss"])
        diff = cand - ref
        differences.append(diff)
        pairs.append(
            {
                "seed": seed,
                "candidate_loss": cand,
                "reference_loss": ref,
                "difference": diff,
            }
        )
    values = np.asarray(differences, dtype=np.float64)
    n = int(values.size)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    if se > 0:
        tstat = mean / se
        pvalue = float(2.0 * stats.t.sf(abs(tstat), df=n - 1))
    else:
        tstat = math.copysign(math.inf, mean) if mean else 0.0
        pvalue = 0.0 if mean else 1.0
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
        "ci_95": [mean - tcrit * se, mean + tcrit * se],
        "wins_candidate_lower": int((values < 0).sum()),
        "ties": int((values == 0).sum()),
        "two_sided_paired_t_pvalue": pvalue,
        "paired_t_statistic": float(tstat),
        "pairs": pairs,
    }


def aggregate(rows: list[dict[str, Any]], optimizer: str) -> dict[str, Any]:
    subset = [row for row in rows if row["optimizer"] == optimizer]
    losses = np.asarray(
        [float(row["final_val_loss"]) for row in subset], dtype=np.float64
    )
    wall = np.asarray([float(row["wall_seconds"]) for row in subset], dtype=np.float64)
    memory = np.asarray(
        [float(row["peak_cuda_gib"]) for row in subset], dtype=np.float64
    )
    return {
        "optimizer": optimizer,
        "mean_final_val_loss": float(losses.mean()),
        "sample_sd_final_val_loss": float(losses.std(ddof=1)),
        "mean_wall_seconds": float(wall.mean()),
        "mean_peak_cuda_gib": float(memory.mean()),
        "provenance": subset[0].get("optimizer_provenance", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report ORBIT-all vs production optimizer baselines"
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--protocol", default="configs/production_optimizer_protocol.json")
    args = parser.parse_args()

    protocol = load_production_protocol(args.protocol)
    work_dir = Path(args.work_dir)
    path = work_dir / "merged" / "production_confirm.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; merge confirmation first")
    rows = json.loads(path.read_text())

    expected = set(protocol["phases"]["confirm"]["seeds"])
    for optimizer in protocol["optimizers"]:
        seeds = {
            int(row["seed"]) for row in rows if row["optimizer"] == optimizer
        }
        if seeds != expected:
            raise RuntimeError(
                f"incomplete held-out seeds for {optimizer}: {sorted(seeds)}"
            )

    references = [
        name for name in protocol["optimizers"] if name != "orbit_all"
    ]
    comparisons = {
        f"orbit_all_vs_{reference}": paired_summary(rows, "orbit_all", reference)
        for reference in references
    }
    aggregates = [aggregate(rows, name) for name in protocol["optimizers"]]
    ranking = sorted(aggregates, key=lambda item: item["mean_final_val_loss"])
    report = {
        "title": "ORBIT-all post-confirmation production optimizer comparison",
        "status": protocol["status"],
        "scope": protocol["interpretation"],
        "comparisons": comparisons,
        "ranking_by_mean_final_validation_loss": ranking,
    }
    analysis_dir = work_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out = analysis_dir / "production_optimizer_comparison.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
