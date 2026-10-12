from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def paired_summary(rows: list[dict[str, Any]], candidate: str, reference: str) -> dict[str, Any]:
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["optimizer"])] = row

    pairs = []
    for seed in sorted(by_seed):
        cell = by_seed[seed]
        if candidate in cell and reference in cell:
            c = float(cell[candidate]["final_val_loss"])
            r = float(cell[reference]["final_val_loss"])
            pairs.append(
                {
                    "seed": seed,
                    "candidate_loss": c,
                    "reference_loss": r,
                    "difference": c - r,
                }
            )
    if len(pairs) < 2:
        raise RuntimeError(f"need at least two paired seeds for {candidate} vs {reference}")

    d = np.asarray([p["difference"] for p in pairs], dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
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
        "ci_95": [mean - tcrit * se, mean + tcrit * se],
        "wins_candidate_lower": int((d < 0).sum()),
        "ties": int((d == 0).sum()),
        "two_sided_paired_t_pvalue": float(p_value),
        "paired_t_statistic": float(t_stat),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare published ORBIT generation with ORBIT-all")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--baseline-config", default="configs/legacy_orbit_paper_recipe.json")
    args = parser.parse_args()

    work = Path(args.work_dir)
    baseline = json.loads(Path(args.baseline_config).read_text())
    confirm_path = work / "merged" / "confirm.json"
    legacy_path = work / "merged" / "legacy_orbit_compare.json"
    if not confirm_path.exists():
        raise FileNotFoundError(f"missing {confirm_path}")
    if not legacy_path.exists():
        raise FileNotFoundError(
            f"missing {legacy_path}; run legacy_orbit_compare.py and merge.py first"
        )

    confirm = json.loads(confirm_path.read_text())
    legacy = json.loads(legacy_path.read_text())
    base_sha = str(baseline["experiment_base_sha"])

    confirm_shas = {row.get("git_sha") for row in confirm}
    if confirm_shas != {base_sha}:
        raise RuntimeError(
            f"confirmation rows do not all come from frozen base SHA {base_sha}: {sorted(confirm_shas)}"
        )
    if {row.get("experiment_base_sha") for row in legacy} != {base_sha}:
        raise RuntimeError("legacy comparison rows do not declare the same frozen experiment base SHA")

    confirm_digests = {row.get("protocol_digest") for row in confirm}
    legacy_digests = {row.get("protocol_digest") for row in legacy}
    if confirm_digests != legacy_digests or len(confirm_digests) != 1:
        raise RuntimeError("confirmation and legacy rows do not share one protocol digest")

    expected_seeds = {int(seed) for seed in baseline["seeds"]}
    legacy_seeds = {int(row["seed"]) for row in legacy}
    if legacy_seeds != expected_seeds:
        raise RuntimeError(
            f"legacy baseline is incomplete: expected={sorted(expected_seeds)}, got={sorted(legacy_seeds)}"
        )
    for optimizer in ("muon", "orbit_qk", "orbit_all"):
        seeds = {int(row["seed"]) for row in confirm if row["optimizer"] == optimizer}
        if seeds != expected_seeds:
            raise RuntimeError(
                f"confirmation optimizer {optimizer} is incomplete: "
                f"expected={sorted(expected_seeds)}, got={sorted(seeds)}"
            )

    rows = confirm + legacy
    comparisons = {
        "orbit_paper_recipe_vs_muon": paired_summary(rows, "orbit_paper_recipe", "muon"),
        "orbit_qk_vs_orbit_paper_recipe": paired_summary(
            rows, "orbit_qk", "orbit_paper_recipe"
        ),
        "orbit_all_vs_orbit_qk": paired_summary(rows, "orbit_all", "orbit_qk"),
        "orbit_all_vs_orbit_paper_recipe": paired_summary(
            rows, "orbit_all", "orbit_paper_recipe"
        ),
    }

    total = comparisons["orbit_all_vs_orbit_paper_recipe"]["mean_difference"]
    retune = comparisons["orbit_qk_vs_orbit_paper_recipe"]["mean_difference"]
    new_families = comparisons["orbit_all_vs_orbit_qk"]["mean_difference"]
    closure_error = total - (retune + new_families)

    report = {
        "title": "ORBIT -> ORBIT-all same-harness generation comparison",
        "experiment_base_sha": base_sha,
        "source_orbit": baseline["source"],
        "paper_recipe": baseline["hparams"],
        "comparisons": comparisons,
        "arithmetic_decomposition": {
            "total_orbit_all_minus_paper_recipe": total,
            "qk_retune_component": retune,
            "beyond_qk_family_component": new_families,
            "closure_error": closure_error,
            "identity": "(ORBIT-all - paper recipe) = (ORBIT-all - retuned ORBIT-QK) + (retuned ORBIT-QK - paper recipe)",
            "caution": "This is an arithmetic same-seed decomposition, not a causal attribution of the gain. Family ablations provide the mechanism evidence.",
        },
        "interpretation_notes": [
            "orbit_paper_recipe is the published ORBIT Q/K recipe transferred into the current frozen ORBIT-all model/data harness.",
            "orbit_qk is the Q/K-only generation independently retuned in the current shared search grid and is the strongest apples-to-apples previous-generation baseline.",
            "orbit_all extends function-aware geometry beyond Q/K; its comparison to orbit_qk isolates the generational extension more cleanly than the paper-recipe comparison.",
            "This study is not an exact reproduction of the historical ORBIT paper stack; the exact historical reproduction remains in the original ORBIT repository.",
        ],
    }

    out_dir = work / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "generation_comparison.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
