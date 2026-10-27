# Production optimizer benchmark

This is a post-confirmation ORBIT-all benchmark against optimizer families that have actually been disclosed in the training of released large language models.

The extension is deliberately separate from the frozen primary campaign. It is based on core commit e804674c1169740d83fc3a13225aa36bf4d16df1, and the campaign runner refuses to run if the frozen geometry/model/optimizer/experiment files differ from that commit.

The current protocol is **v1**. A v0 tuning-only pilot was completed on seed 900 and then stopped before any held-out production-confirmation seed was evaluated. That pilot exposed an inappropriate base-learning-rate range for RMS-scaled Moonlight/Kimi Muon, a missing Nesterov look-ahead in the same-harness Moonshot implementation, a top-boundary PaLM Adafactor winner, and an Adam grid that confounded learning rate with weight decay. v1 repairs those issues while keeping ORBIT-all frozen. The exact v0 protocol is archived at `configs/production_optimizer_protocol_v0.json`.

## What the benchmark asks

On the same 124M-class decoder, FineWeb-Edu corpus, token budget, paired initialization/data streams, validation procedure, and common learning-rate schedule, how does ORBIT-all compare with optimizer algorithms that have been used for real released LLMs?

This is not an exact reproduction of Kimi K2, Moonlight, Qwen, DeepSeek, PaLM, MPT, or GPT-NeoX. Those systems differ in architecture, scale, data, batch schedule, precision, distributed implementation, and training recipe. Lab/model names establish optimizer provenance only.

Because this benchmark was designed after the main ORBIT-all confirmation, it must be reported as post-confirmation secondary evidence.

## Production provenance

- muonclip_kimi — Kimi Team / Moonshot AI — Kimi K2 — MuonClip.
  Source: https://arxiv.org/abs/2507.20534
- muon_moonlight — Moonshot AI — Moonlight-16B-A3B — scaled Muon with weight decay and consistent update RMS.
  Source: https://github.com/MoonshotAI/Moonlight
- adamw — Alibaba Qwen and DeepSeek — Qwen-7B and DeepSeek LLM 7B/67B.
  Sources: https://github.com/QwenLM/Qwen/blob/main/tech_memo.md and https://arxiv.org/abs/2401.02954
- lion — MosaicML / Databricks — MPT-7B.
  Source: https://www.databricks.com/blog/mpt-7b
- adafactor_palm — Google Research — PaLM 8B/62B/540B — unfactorized Adafactor with parameter scaling.
  Source: https://arxiv.org/abs/2204.02311
- adam — EleutherAI — GPT-NeoX-20B.
  Source: https://github.com/EleutherAI/gpt-neox/blob/main/configs/20B.yml

Kimi K2 uses MLA. The benchmark model uses ordinary MHA, so muonclip_kimi implements a regular-MHA adaptation of the K2 report's per-head QK-Clip rule rather than claiming an MLA reproduction. It uses tau=100 on top of the corrected Moonshot-style base: momentum, recommended Nesterov look-ahead, Newton-Schulz, the Moonshot 0.2 * sqrt(max(m,n)) update-RMS scaling, weight decay, and post-update Q/K clipping.

PaLM reports Adafactor without factorization, beta1=0.9, beta2=1-k^-0.8, and parameter-RMS learning-rate scaling. The benchmark keeps those optimizer mechanics but intentionally uses the common benchmark schedule rather than PaLM's entire training recipe.

## Fairness protocol

Optimizer comparisons are highly sensitive to tuning. The benchmark therefore uses optimizer-specific tuning rather than transferring ORBIT-all's learning rate to competitors.

- ORBIT-all gets no new post-hoc tuning. It is frozen to the earlier shared-03 recipe: matrix_lr 0.018, aux_lr 0.0085, weight_decay 0.0045.
- Each production baseline gets a predeclared nine-candidate v1 grid on tuning seed 900.
- Moonlight/Kimi v1 candidates use the same base learning rate for the matrix and auxiliary Adam-style paths, matching the public Moonlight toy implementation more closely.
- Learning-rate sweeps hold weight decay fixed for the main bracket; separate candidates probe weight-decay sensitivity near the center instead of entangling both dimensions on every point.
- Winners are frozen before confirmation.
- Held-out confirmation uses seeds 910 through 919 for all seven optimizers.
- Every method gets the same 900-step budget, data pairing, evaluation, and common 5% warmup plus cosine-to-10% schedule.
- Final validation loss, paired win counts, paired Student-t intervals, wall time, and peak CUDA memory are reported.

The design follows the general lesson from systematic optimizer benchmarking that blind hyperparameter transfer can reverse optimizer rankings and make comparisons unfair.

## Work directories

Do not write these runs into the primary ORBIT-all evidence directory.

Recommended environment variables:

    export ORBIT_ALL_DATA_WORK=/content/drive/MyDrive/orbit_all_paper
    export ORBIT_ALL_PROD_WORK=/content/drive/MyDrive/orbit_all_production_optimizers_v1

The first directory supplies the already-prepared frozen train.bin and val.bin. The second stores this benchmark only.

## Smoke

    python scripts/production_optimizer_campaign.py --phase smoke \
      --work-dir "$ORBIT_ALL_PROD_WORK" \
      --data-work-dir "$ORBIT_ALL_DATA_WORK" \
      --device cpu --steps-override 2

Smoke output is synthetic plumbing only.

## Tuning

The v1 tuning phase contains **55 runs**: one frozen ORBIT-all recipe and nine candidates for each of six production baselines.

Run four workers using shard-id 0, 1, 2, and 3:

    python scripts/production_optimizer_campaign.py --phase tune \
      --work-dir "$ORBIT_ALL_PROD_WORK" \
      --data-work-dir "$ORBIT_ALL_DATA_WORK" \
      --shard-id 0 --num-shards 4

After every shard is complete:

    python scripts/production_optimizer_merge.py --phase tune \
      --work-dir "$ORBIT_ALL_PROD_WORK"

This creates and freezes merged/production_best_configs.json. Inspect all candidate losses before confirmation. If any v1 winner still sits on an obviously improving outer LR boundary, extend that baseline's bracket in a new protocol revision before touching seeds 910-919.

## Held-out confirmation

Confirmation contains 70 runs: seven optimizers times ten held-out seeds.

Again run four shards:

    python scripts/production_optimizer_campaign.py --phase confirm \
      --work-dir "$ORBIT_ALL_PROD_WORK" \
      --data-work-dir "$ORBIT_ALL_DATA_WORK" \
      --shard-id 0 --num-shards 4

Then:

    python scripts/production_optimizer_merge.py --phase confirm \
      --work-dir "$ORBIT_ALL_PROD_WORK"

    python scripts/production_optimizer_report.py \
      --work-dir "$ORBIT_ALL_PROD_WORK"

The final report contains ORBIT-all minus every production optimizer on paired held-out seeds and a ranking by mean final validation loss.

## Claim discipline

A positive result does not mean ORBIT-all is better than Kimi K2, Qwen, DeepSeek, PaLM, MPT, or GPT-NeoX as models.

The defensible claim is narrower: in this controlled 124M-class pretraining harness, ORBIT-all outperformed same-harness implementations of optimizer algorithms disclosed for those production model families.

Scaling that conclusion to billion- or trillion-parameter training remains a separate empirical question.
