# ORBIT-all ablation plan

The ablation campaign is designed to distinguish three different questions: whether ORBIT-all works, which functional family causes the effect, and whether the effect transfers to modern architecture variants.

## Core mechanism ablations

Use one frozen ORBIT-all recipe across these variants:

```text
orbit_qk
orbit_qk_vo
orbit_qk_vo_swiglu
orbit_all
orbit_all_minus_qk
orbit_all_minus_vo
orbit_all_minus_swiglu
orbit_all_minus_embedding
```

The add-family sequence measures incremental benefit. The drop-family sequence tests whether a family remains useful after the other geometries are present.

## Q/K-specific sub-ablations

After the top-level family campaign, add targeted Q/K ablations only if Q/K remains an important component:

```text
opposite-side vs same-side statistics
rotary transport vs no rotary transport
full 2x2 frequency metric vs diagonal
GQA group aggregation vs naive repeated-K approximation
post-QK-norm statistics vs pre-normalization statistics
joint Q/K norm restoration vs independent restoration
```

The original ORBIT paper already motivates several of these; ORBIT-all should avoid rerunning expensive variants unless the new architecture changes the scientific question.

## V/O sub-ablations

The initial V sensitivity is a content-transport proxy based on context-to-value second-moment ratios. Compare it against:

```text
identity / no V-O geometry
O-only input-second-moment geometry
V-only transport geometry
V+O with joint restoration
head-scalar V metric instead of channelwise V metric
```

This is especially important because V/O is less directly derived than the Q/K and SwiGLU local sensitivities.

## SwiGLU sub-ablations

The strongest mechanistic tests are:

```text
full derivative-aware gate metric
replace gate metric with activation variance only
up-only geometry
new down-input geometry only
independent norm restoration vs joint gate/up/down restoration
```

The derivative-aware comparison tests whether the actual multiplicative function matters beyond ordinary activation statistics.

## Embedding/output sub-ablations

The first implementation uses token event frequency. If it contributes materially, test:

```text
input-frequency only
output-target frequency only
combined frequency
softmax-Fisher row metric
untied input/output embeddings with separate geometries
```

Do not spend large compute on these unless the top-level embedding drop ablation shows an effect.

## Learned-RoPE ablations

For learned log frequencies:

```text
AdamW only
ORBIT-all frequency metric
frequency metric without delta^2 factor
frequency metric without Q/K second moments
log-frequency parameterization vs direct positive-frequency parameterization
```

This section is transfer/mechanism evidence for a special structural parameter, not a justification to specialize arbitrary vectors.

## MoE ablations

Separate router and expert questions:

```text
Muon router + ordinary experts
categorical-Fisher router only
routed-token SwiGLU experts only
router + expert geometry
router geometry with/without load diagnostic conditioning (future)
```

Expert statistics are already conditional on routed tokens; that fact should be preserved when comparing against dense MLP geometry.

## Efficiency ablations

Every major family should report its incremental training overhead and optimizer-state memory. A family that provides a statistically tiny improvement at a large systems cost may be scientifically interesting but should not be presented as a practical default.

## Stopping rule

Do not run every sub-ablation by default. First run the top-level family campaign. Expand only the families that show measurable contribution or surprising failure. This keeps the project focused on falsifying meaningful hypotheses rather than generating a combinatorial grid.
