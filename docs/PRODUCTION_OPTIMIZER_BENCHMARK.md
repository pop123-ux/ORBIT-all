# Production optimizer benchmark

This is a post-confirmation ORBIT-all benchmark against optimizer families that have actually been disclosed in the training of released large language models.

The extension is deliberately separate from the frozen primary campaign. It is based on core commit e804674c1169740d83fc3a13225aa36bf4d16df1, and the campaign runner refuses to run if the frozen geometry/model/optimizer/experiment files differ from that commit.

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

Kimi K2 uses MLA. The benchmark model uses ordinary MHA, so muonclip_kimi implements the K2 report's regular-MHA per-head QK-Clip rule rather than claiming an MLA reproduction. It uses tau=100, momentum Muon, Newton-Schulz, the Moonshot 0.2 * sqrt(max(m,n)) RMS-matching scale, and post-update Q/K clipping.

PaLM reports Adafactor without factorization, beta1=0.9, beta2=1-k^-0.8, and parameter-RMS learning-rate scaling. The benchmark keeps those optimizer mechanics but intentionally uses the common benchmark schedule rather than PaLM's entire training recipe.

## Fairness protocol

Optimizer comparisons are highly sensitive to tuning. The benchmark therefore uses optimizer-specific tuning rather than transferring ORBIT-all's learning rate to competitors.

- ORBIT-all gets no new post-hoc tuning. It is frozen to the earlier shared-03 recipe: matrix_lr 0.018, aux_lr 0.0085, weight_decay 0.0045.
- Each production baseline gets a predeclared five-candidate grid on tuning seed 900.
- Winners are frozen before confirmation.
- Held-out confirmation uses seeds 910 through 919 for all seven optimizers.
- Every method gets the same 900-step budget, data pairing, evaluation, and common 5% warmup plus cosine-to-10% schedule.
- Final validation loss, paired win counts, paired Student-t intervals, wall time, and peak CUDA memory are reported.

The design follows the general lesson from systematic optimizer benchmarking that blind hyperparameter transfer can reverse optimizer rankings and make comparisons unfair.

## Work directories

Do not write these runs into the primary ORBIT-all evidence directory.

Recommended environment variables:

    export ORBIT_ALL_DATA_WORK=/content/drive/MyDrive/orbit_all_paper
    export ORBIT_ALL_PROD_WORK=/content/drive/MyDrive/orbit_all_production_optimizers

The first directory supplies the already-prepared frozen train.bin and val.bin. The second stores this benchmark only.

## Smoke

    python scripts/production_optimizer_campaign.py --phase smoke \
      --work-dir "$ORBIT_ALL_PROD_WORK" \
      --data-work-dir "$ORBIT_ALL_DATA_WORK" \
      --device cpu --steps-override 2

Smoke output is synthetic plumbing only.

## Tuning

The tuning phase contains 31 runs: one frozen ORBIT-all recipe and five candidates for each of six production baselines.

Run four workers using shard-id 0, 1, 2, and 3:

    python scripts/production_optimizer_campaign.py --phase tune \
      --work-dir "$ORBIT_ALL_PROD_WORK" \
      --data-work-dir "$ORBIT_ALL_DATA_WORK" \
      --shard-id 0 --num-shards 4

After every shard is complete:

    python scripts/production_optimizer_merge.py --phase tune \
      --work-dir "$ORBIT_ALL_PROD_WORK"

This creates and freezes merged/production_best_configs.json.

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
