# ORBIT-all

**Function-aware optimization across the major matrix families of modern language models.**

ORBIT-all extends ORBIT beyond rotary query-key optimization. Instead of assigning optimizer geometry from tensor shape alone, ORBIT-all assigns geometry from the computation performed by a major parameter family.

The initial research scope is deliberately limited to large, structurally distinct matrix families:

- query/key score geometry, including GQA and QK normalization;
- value/output content-flow geometry;
- gated MLP geometry for SwiGLU-style gate/up/down projections;
- embedding and language-model-head geometry;
- MoE router and expert geometry;
- learned or alternative rotary parameterizations;
- longer-context transfer.

Biases, normalization vectors, and ordinary scalars remain on a conventional auxiliary AdamW path unless a later experiment provides a strong functional reason to specialize them.

## Research thesis

A parameter tensor's shape describes how it is stored. Its computational role describes how changing it changes the model. ORBIT-all tests whether optimizer geometry should follow the second.

The project therefore studies the progression

```text
scalar-aware optimization
        -> matrix-aware optimization
        -> function-aware optimization
```

## Experimental philosophy

The repository is organized as a paper campaign rather than a collection of notebooks. Expensive experiments are gated by correctness checks and cheaper falsification experiments. Held-out evaluation is frozen before results are inspected. Every run carries code/config/environment provenance. Negative runs are retained. Paper figures and statistics must be regenerated from measured artifacts rather than hand-entered values.

See:

- `docs/SCIENTIFIC_CONTRACT.md` for non-negotiable experimental rules;
- `docs/METHOD.md` for the initial geometry definitions;
- `docs/EXPERIMENT_PROTOCOL.md` for the staged campaign;
- `docs/ABLATIONS.md` for the planned mechanism and transfer studies;
- `docs/RUNBOOK.md` for runnable commands.

## Install

```bash
git clone https://github.com/pop123-ux/ORBIT-all.git
cd ORBIT-all
python -m pip install -e .
```

Developer install:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Fast smoke test

```bash
python scripts/campaign.py \
  --phase smoke \
  --work-dir /tmp/orbit_all_smoke \
  --device cpu \
  --synthetic \
  --steps-override 2
```

The smoke phase is plumbing-only. It is not evidence for the paper.

## Planned campaign

```text
0. correctness + synthetic smoke
1. dense discovery
2. matched tuning: Muon vs ORBIT vs ORBIT-all
3. held-out paired confirmation
4. add-family and drop-family mechanism ablations
5. GQA transfer
6. QK-normalization transfer
7. rotary-parameterization transfer
8. MoE transfer
9. longer-context transfer
10. long-horizon confirmation
11. larger-model transfer
```

The exact protocol lives in `configs/protocol.json` and is treated as the machine-readable research contract.

## Status

Early research implementation. Geometry choices are hypotheses until they survive the predeclared ablations and held-out experiments.
