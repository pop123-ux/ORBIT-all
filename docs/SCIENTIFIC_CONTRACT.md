# ORBIT-all scientific contract

This file is part of the experimental protocol. Change it before a phase starts, never after seeing that phase's held-out results.

## What ORBIT-all is testing

The primary hypothesis is not that every parameter deserves a custom adaptive statistic. The hypothesis is narrower:

> Major LLM matrix families with different computational roles can benefit from different optimizer geometries, even when they share the same tensor rank.

The initial specialized families are Q/K, V/O, gated MLP projections, token embedding/output weights, learned rotary frequencies, and MoE routing/expert blocks. Biases, normalization vectors, and ordinary scalars remain conventional auxiliary parameters.

## Evidence rules

1. **Measured results only.** Figures, tables, confidence intervals and manuscript macros must be generated from measured result files. If a required measurement is missing, artifact generation must fail rather than invent a placeholder value.
2. **No silent recipe changes.** Matched tuning freezes one recipe per optimizer before held-out confirmation. If optimizer equations change afterward, start a new work directory and retune.
3. **No mixed revisions.** The merger rejects phases containing multiple Git SHAs or protocol digests.
4. **Keep negative runs.** Failed hypotheses and losing seeds are part of the record. Do not delete them to improve the story.
5. **Synthetic runs are never paper evidence.** They test code paths only.
6. **Matched comparisons are paired by seed.** The primary estimand is final validation loss difference on the same seed and architecture.
7. **Primary comparison is declared in advance.** ORBIT-all minus Muon on the dense held-out confirmation is primary. ORBIT-all minus QK-only ORBIT and mechanism/transfer ablations are secondary.
8. **Uncertainty is reported.** For paired seed differences, report mean, sample SD, win count and a two-sided 95% Student-t confidence interval. Do not hide the raw paired differences.
9. **Transfer is not mechanism proof.** GQA, QK-norm, learned-RoPE, MoE and context experiments test transfer. Family add/drop ablations test mechanism.
10. **No inference claims from training-only evidence.** ORBIT-all is designed for zero inference overhead, but training improvements do not imply inference quality or downstream task gains without separate evidence.

## Claim discipline

A family is considered supported only if its removal or addition changes the result consistently enough to survive seed noise. If a drop-family ablation is indistinguishable from full ORBIT-all, the paper must weaken or remove the corresponding mechanism claim.

The learned geometries are optimizer designs motivated by local functional sensitivity. Unless explicitly derived and verified otherwise, they are not to be described as exact Fisher, Hessian, natural-gradient, or exact function-space pullbacks.

## Before public release

- rerun the prior-art search for the exact equations and family-specific constructions;
- verify all paper numbers against merged artifacts;
- regenerate every figure from raw measured results;
- run the full correctness test suite;
- archive the exact protocol, code commit, dataset revision, tokenizer, package versions, seeds and frozen recipes used in the paper.
