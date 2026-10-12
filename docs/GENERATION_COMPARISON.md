# ORBIT -> ORBIT-all generation comparison

This study compares the previous ORBIT generation with ORBIT-all without weakening the main matched-confirmation protocol.

## Why there are two ORBIT baselines

The current ORBIT-all confirmation already includes `orbit_qk`, the Q/K-only function-aware optimizer independently tuned on the same shared grid as Muon and ORBIT-all. In the dense MHA setting used for confirmation, its Q/K metric, opposite-side rotary transport, inverse 2x2 metric power, Muon candidate update, auxiliary AdamW path, and joint Q/K Frobenius restoration match the core algorithmic structure of the original ORBIT generation.

A second baseline, `orbit_paper_recipe`, transfers the exact published ORBIT paper recipe into the current frozen ORBIT-all harness:

- matrix learning rate: `0.0181771645661641`
- auxiliary learning rate: `0.008917991355665525`
- weight decay: `0.005835070036773646`
- momentum: `0.95`
- Newton-Schulz steps: `5`
- functional power: `1.0`
- metric epsilon: `1e-5`
- metric condition cap: `100`
- deltas: `[1, 2, 4, 8, 16, 32, 64, 128]`

The source is `pop123-ux/ORBIT` at commit `a6abbfc22cfcb3f874ce7bac6b36efebd3105629`, config `configs/paper_124m_matched.json`, selected candidate `shared-04`.

## What each comparison answers

`orbit_paper_recipe` vs `muon`

: How the published ORBIT recipe transfers into the newer frozen dense model/data harness.

`orbit_qk` vs `orbit_paper_recipe`

: How much changes when the same Q/K-only generation is retuned under the current shared search protocol.

`orbit_all` vs `orbit_qk`

: The cleanest same-harness generational comparison. This measures the increment from extending function-aware optimization beyond Q/K while using the current independently selected recipe.

`orbit_all` vs `orbit_paper_recipe`

: Total same-harness improvement from the published ORBIT recipe to ORBIT-all.

The report also prints the arithmetic identity

```text
(ORBIT-all - paper recipe)
  = (ORBIT-all - retuned ORBIT-QK)
  + (retuned ORBIT-QK - paper recipe)
```

This decomposition is descriptive, not causal. The family-ablation campaign is the mechanism test.

## Important interpretation boundary

`orbit_paper_recipe` is **not** an exact historical ORBIT paper reproduction. The model architecture, token pool, validation pool, and current harness remain the frozen ORBIT-all setup so that losses are directly comparable. The exact historical paper-stack reproduction remains in the original ORBIT repository.

This distinction should remain explicit in the paper:

- use the original ORBIT repository for historical reproducibility;
- use `orbit_paper_recipe` for recipe transfer into the new harness;
- use `orbit_qk` as the strongest apples-to-apples previous-generation baseline;
- use family ablations to explain where ORBIT-all's additional gain comes from.

## Provenance gate

The comparison runner refuses to execute if any of these frozen experiment-core files differ from commit `e804674c1169740d83fc3a13225aa36bf4d16df1`:

```text
src/orbit_all/geometry.py
src/orbit_all/model.py
src/orbit_all/optimizer.py
src/orbit_all/experiment.py
configs/protocol.json
```

It also verifies the frozen tuning SHA, confirmation SHA, protocol digest, pinned dataset revision, and `paper_mode=true` data manifest before running.

## Commands

After the current family-ablation campaign is complete, check out branch `legacy-orbit-generation-compare` and use the same persistent `ORBIT_ALL_WORK` directory.

Run the ten paper-recipe transfer seeds across four workers:

```bash
python scripts/legacy_orbit_compare.py --work-dir "$ORBIT_ALL_WORK" --shard-id 0 --num-shards 4
python scripts/legacy_orbit_compare.py --work-dir "$ORBIT_ALL_WORK" --shard-id 1 --num-shards 4
python scripts/legacy_orbit_compare.py --work-dir "$ORBIT_ALL_WORK" --shard-id 2 --num-shards 4
python scripts/legacy_orbit_compare.py --work-dir "$ORBIT_ALL_WORK" --shard-id 3 --num-shards 4
```

Merge the baseline:

```bash
python scripts/merge.py --phase legacy_orbit_compare --work-dir "$ORBIT_ALL_WORK"
```

Build the generation report:

```bash
python scripts/generation_compare.py --work-dir "$ORBIT_ALL_WORK"
```

The report is written to:

```text
$ORBIT_ALL_WORK/analysis/generation_comparison.json
```
