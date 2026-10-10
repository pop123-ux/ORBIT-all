# ORBIT-all experiment protocol

`configs/protocol.json` is the machine-readable source of truth. This document explains why the phases are separated.

## Phase 0 — correctness and smoke

Run unit tests, then a two-step synthetic CPU/GPU smoke. This catches routing, GQA shape, learned-RoPE, MoE and optimizer-state errors without producing scientific evidence.

## Phase 1 — cheap discovery

Dense 124M-class model, 300 steps, one development seed. Muon, QK-only ORBIT and ORBIT-all use the same sparse subset of the predeclared candidate grid. This phase is only for detecting catastrophic learning-rate regions and implementation pathologies.

No headline result should come from discovery.

## Phase 2 — matched tuning

Each of Muon, QK-only ORBIT and ORBIT-all receives the exact same ten candidate recipes at 900 steps. No optimizer-specific extra tuning dimensions are allowed. Select the minimum validation-loss candidate independently for each optimizer and freeze it.

Once `best_configs.json` exists, do not edit it. If optimizer equations change, start a new work directory and rerun matched tuning.

## Phase 3 — held-out paired confirmation

Evaluate all three frozen recipes on seeds 500–509. The primary paper estimand is

```text
final validation loss(ORBIT-all) - final validation loss(Muon)
```

computed per matched seed.

Report:

```text
paired mean difference
sample SD of paired differences
95% Student-t confidence interval
win count
wall-clock overhead
peak allocated CUDA memory
```

ORBIT-all minus QK-only ORBIT is an important secondary contrast because it asks whether the extension beyond Q/K adds measurable value.

## Phase 4 — family mechanism ablations

Use the frozen ORBIT-all recipe for all family ablations so learning-rate retuning cannot hide which geometry matters.

Two complementary sequences are used:

```text
ADD FAMILY
ORBIT-QK
  -> + V/O
  -> + SwiGLU
  -> full ORBIT-all

DROP FAMILY
full ORBIT-all
  -> minus Q/K
  -> minus V/O
  -> minus SwiGLU
  -> minus embedding geometry
```

If a family disappears without measurable degradation, the paper must not claim that family as necessary.

## Phase 5 — architecture transfer

Transfer experiments are intentionally separate from mechanism ablations:

```text
GQA
QK normalization
learned rotary frequencies
MoE
```

For each architecture, compare Muon, QK-only ORBIT and ORBIT-all under the frozen dense recipes first. Only if transfer clearly fails should a separately declared architecture-specific retuning phase be added.

This prevents optimizer-specific retuning from being confused with architectural robustness.

## Phase 6 — longer context

Repeat the three-way comparison at 1024 and 2048 tokens. Batch size may be reduced and gradient accumulation increased to preserve an approximately matched tokens-per-update budget. Any such change must be encoded in the protocol before results are collected.

## Phase 7 — long horizon

Run 2700-step dense confirmation on a smaller set of held-out seeds. This is a transfer check for persistence of the effect, not a replacement for the ten-seed primary comparison.

## Larger-model transfer

A larger model phase is intentionally not hard-coded yet. Add it only after the method survives dense confirmation, mechanism ablations and architecture transfer. The larger configuration must be frozen in the protocol before its first result is observed.

## Statistical hierarchy

The sole initial primary confirmatory comparison is ORBIT-all versus Muon on the dense held-out confirmation. Other intervals are secondary. If later versions promote multiple architecture-specific claims to confirmatory status, add a multiplicity plan before running those experiments.

The primary interval is a two-sided 95% Student-t interval over paired seed differences because the population variance is unknown and estimated from a small number of paired runs.
