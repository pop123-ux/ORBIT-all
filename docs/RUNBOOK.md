# ORBIT-all runbook

This runbook follows the same staged, sharded workflow used for the earlier ORBIT paper campaign. Keep one shared work directory across GPU workers.

## 0. Clone and install

On every worker:

```bash
git clone https://github.com/pop123-ux/ORBIT-all.git
cd ORBIT-all
python -m pip install -q -e ".[data,dev]"
pytest -q
```

If the repository is private, authenticate Git without printing or committing your token. Do not store a personal access token inside a notebook cell that will be shared.

Record the revision before starting a phase:

```bash
git rev-parse HEAD
```

Do not pull new optimizer code into only some workers mid-phase. The merger rejects mixed Git SHAs.

## 1. CPU smoke before spending GPU time

```bash
python scripts/campaign.py \
  --phase smoke \
  --work-dir /tmp/orbit_all_smoke \
  --device cpu \
  --synthetic \
  --steps-override 2
```

Then run one short GPU smoke:

```bash
python scripts/campaign.py \
  --phase smoke \
  --work-dir /tmp/orbit_all_gpu_smoke \
  --device cuda \
  --synthetic \
  --steps-override 5
```

These results are plumbing checks only.

## 2. Pin and prepare FineWeb-Edu

Resolve an immutable Hugging Face dataset revision before any evidence run:

```bash
python - <<'PY'
from huggingface_hub import HfApi
info = HfApi().repo_info("HuggingFaceFW/fineweb-edu", repo_type="dataset")
print(info.sha)
PY
```

Put that SHA into `configs/protocol.json` under `data.revision`, commit the change, and use the same commit on every worker.

For a shared Google Drive directory in Colab, for example:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
export ORBIT_ALL_WORK=/content/drive/MyDrive/orbit_all_paper
```

Prepare the token files once:

```bash
python scripts/prepare_data.py \
  --work-dir "$ORBIT_ALL_WORK" \
  --revision YOUR_PINNED_DATASET_SHA \
  --train-tokens 50000000 \
  --val-tokens 2000000 \
  --paper-mode
```

Every worker then reads the same `train.bin`, `val.bin`, and `manifest.json`.

## 3. Cheap discovery

With four GPU workers, use shard IDs 0, 1, 2, 3 respectively:

```bash
python scripts/campaign.py \
  --phase discovery \
  --work-dir "$ORBIT_ALL_WORK" \
  --shard-id 0 \
  --num-shards 4
```

Before launching, inspect ownership without running:

```bash
python scripts/campaign.py \
  --phase discovery \
  --work-dir "$ORBIT_ALL_WORK" \
  --shard-id 0 \
  --num-shards 4 \
  --list-tasks
```

Repeat with the other shard IDs. When all workers finish:

```bash
python scripts/merge.py --phase discovery --work-dir "$ORBIT_ALL_WORK"
```

Discovery is a sanity gate, not the paper headline.

## 4. Matched tuning

Run the exact same ten-candidate grid for Muon, QK-only ORBIT and ORBIT-all:

```bash
python scripts/campaign.py \
  --phase matched_tune \
  --work-dir "$ORBIT_ALL_WORK" \
  --shard-id 0 \
  --num-shards 4
```

Use shard IDs 0–3 across workers, then freeze recipes:

```bash
python scripts/merge.py --phase matched_tune --work-dir "$ORBIT_ALL_WORK"
cat "$ORBIT_ALL_WORK/merged/best_configs.json"
```

Once this file exists, do not edit it after seeing confirmation results. If the optimizer implementation changes, use a new work directory and retune.

## 5. Ten-seed held-out confirmation

```bash
python scripts/campaign.py \
  --phase confirm \
  --work-dir "$ORBIT_ALL_WORK" \
  --shard-id 0 \
  --num-shards 4
```

After all shards:

```bash
python scripts/merge.py --phase confirm --work-dir "$ORBIT_ALL_WORK"
python scripts/analyze.py \
  --phase confirm \
  --candidate orbit_all \
  --reference muon \
  --work-dir "$ORBIT_ALL_WORK"

python scripts/analyze.py \
  --phase confirm \
  --candidate orbit_all \
  --reference orbit_qk \
  --work-dir "$ORBIT_ALL_WORK"
```

Do not escalate because one seed wins. Inspect the paired mean, SD, 95% Student-t interval, win count, runtime and memory.

## 6. Family ablations

Only after the confirmation result is stable:

```bash
python scripts/campaign.py \
  --phase family_ablation \
  --work-dir "$ORBIT_ALL_WORK" \
  --shard-id 0 \
  --num-shards 4
```

Then:

```bash
python scripts/merge.py --phase family_ablation --work-dir "$ORBIT_ALL_WORK"
```

Use the frozen ORBIT-all recipe for all family variants. Do not retune each ablation.

## 7. Architecture transfer

Run only the architecture phases justified by the earlier gates:

```bash
for PHASE in gqa_transfer qknorm_transfer rope_transfer moe_transfer; do
  python scripts/campaign.py \
    --phase "$PHASE" \
    --work-dir "$ORBIT_ALL_WORK" \
    --shard-id 0 \
    --num-shards 4
done
```

Again split shard IDs across workers and merge each phase separately.

## 8. Longer context and horizon

```bash
python scripts/campaign.py --phase context_transfer --work-dir "$ORBIT_ALL_WORK" --shard-id 0 --num-shards 4
python scripts/campaign.py --phase horizon --work-dir "$ORBIT_ALL_WORK" --shard-id 0 --num-shards 4
```

Long-context settings are more memory-sensitive. Confirm the final batch/accumulation policy in `configs/protocol.json` before treating those runs as evidence.

## Useful development commands

One task only:

```bash
python scripts/campaign.py \
  --phase discovery \
  --work-dir /tmp/orbit_all_dev \
  --synthetic \
  --steps-override 10 \
  --max-tasks 1
```

List tasks assigned to a worker:

```bash
python scripts/campaign.py --phase matched_tune --work-dir "$ORBIT_ALL_WORK" --shard-id 2 --num-shards 4 --list-tasks
```

Run tests after every optimizer equation change:

```bash
pytest -q
```

## Non-negotiable restart rule

A code change to optimizer equations, statistics, model semantics, data split, scheduler, or evaluation logic invalidates an in-progress phase. Commit the change, create a fresh work directory for that phase/campaign, and rerun. Do not merge old and new rows.
