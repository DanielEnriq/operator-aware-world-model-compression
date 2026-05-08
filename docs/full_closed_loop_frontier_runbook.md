# Full Closed-Loop Frontier Runbook

This runbook executes the Prompt 1 paper pipeline:
- teacher anchor
- baseline compression (`weight_svd`, `activation_svd`)
- operator-aware distillation (`operator_cost_kl`, `operator_hybrid`, optional `operator_elite`)
- closed-loop benchmarking via `scripts/benchmark_cost_model.py` only

Closed-loop success is the primary reported metric. Operator cache is training signal for operator-aware methods only.

## Conservative Full Runs (n=64, distill_steps=1000)

### A) LeWM TwoRoom

```bash
uv run python scripts/run_full_closed_loop_frontier.py \
  --env tworoom \
  --model-family lewm_hf \
  --teacher-checkpoint quentinll/lewm-tworooms \
  --device cuda \
  --ranks "0.90,0.80,0.70,0.60,0.50" \
  --methods "weight_svd,activation_svd,operator_cost_kl,operator_hybrid" \
  --operator-init weight_svd \
  --num-eval 64 \
  --seed 0 \
  --compress-if-missing \
  --distill-if-missing \
  --build-train-cache-if-missing \
  --skip-existing

uv run python scripts/summarize_full_closed_loop_frontier.py \
  --input-json outputs/tables/full_closed_loop_frontier_all.json \
  --env tworoom \
  --model-family lewm_hf \
  --num-eval 64 \
  --seed 0

uv run python scripts/plot_full_closed_loop_frontier.py \
  --summary-json outputs/tables/full_closed_loop_frontier_tworoom_lewm_hf.json \
  --env tworoom \
  --model-family lewm_hf \
  --num-eval 64 \
  --seed 0 \
  --write-pdf
```

### B) LeWM PushT

```bash
uv run python scripts/run_full_closed_loop_frontier.py \
  --env pusht \
  --model-family lewm_hf \
  --teacher-checkpoint quentinll/lewm-pusht \
  --device cuda \
  --ranks "0.90,0.80,0.70,0.60,0.50" \
  --methods "weight_svd,activation_svd,operator_cost_kl,operator_hybrid" \
  --operator-init weight_svd \
  --num-eval 64 \
  --seed 0 \
  --compress-if-missing \
  --distill-if-missing \
  --build-train-cache-if-missing \
  --skip-existing

uv run python scripts/summarize_full_closed_loop_frontier.py \
  --input-json outputs/tables/full_closed_loop_frontier_all.json \
  --env pusht \
  --model-family lewm_hf \
  --num-eval 64 \
  --seed 0

uv run python scripts/plot_full_closed_loop_frontier.py \
  --summary-json outputs/tables/full_closed_loop_frontier_pusht_lewm_hf.json \
  --env pusht \
  --model-family lewm_hf \
  --num-eval 64 \
  --seed 0 \
  --write-pdf
```

### C) SWM PreJEPA TwoRoom

```bash
uv run python scripts/run_full_closed_loop_frontier.py \
  --env tworoom \
  --model-family swm_prejepa_local \
  --teacher-checkpoint checkpoints/swm_prejepa/tworoom/oawc_swm_prejepa_tworoom \
  --device cuda \
  --ranks "0.90,0.80,0.70,0.60,0.50" \
  --methods "weight_svd,activation_svd,operator_cost_kl,operator_hybrid" \
  --operator-init weight_svd \
  --target-substrings "predictor" \
  --num-eval 64 \
  --seed 0 \
  --compress-if-missing \
  --distill-if-missing \
  --build-train-cache-if-missing \
  --skip-existing

uv run python scripts/summarize_full_closed_loop_frontier.py \
  --input-json outputs/tables/full_closed_loop_frontier_all.json \
  --env tworoom \
  --model-family swm_prejepa_local \
  --num-eval 64 \
  --seed 0

uv run python scripts/plot_full_closed_loop_frontier.py \
  --summary-json outputs/tables/full_closed_loop_frontier_tworoom_swm_prejepa_local.json \
  --env tworoom \
  --model-family swm_prejepa_local \
  --num-eval 64 \
  --seed 0 \
  --write-pdf
```

### D) SWM PreJEPA PushT (placeholder if checkpoint exists)

```bash
uv run python scripts/run_full_closed_loop_frontier.py \
  --env pusht \
  --model-family swm_prejepa_local \
  --teacher-checkpoint checkpoints/swm_prejepa/pusht/<checkpoint_dir> \
  --device cuda \
  --ranks "0.90,0.80,0.70,0.60,0.50" \
  --methods "weight_svd,activation_svd,operator_cost_kl,operator_hybrid" \
  --operator-init weight_svd \
  --target-substrings "predictor" \
  --num-eval 64 \
  --seed 0 \
  --compress-if-missing \
  --distill-if-missing \
  --build-train-cache-if-missing \
  --skip-existing
```

## Smoke Runs (n=8, rank=0.80, distill_steps=50)

```bash
uv run python scripts/run_full_closed_loop_frontier.py \
  --env tworoom \
  --model-family lewm_hf \
  --teacher-checkpoint quentinll/lewm-tworooms \
  --device cuda \
  --ranks "0.80" \
  --methods "weight_svd,operator_hybrid" \
  --operator-init weight_svd \
  --num-eval 8 \
  --seed 0 \
  --distill-steps 50 \
  --compress-if-missing \
  --distill-if-missing \
  --build-train-cache-if-missing
```

## Four-Notebook Bundling

```bash
uv run python scripts/export_result_bundle.py \
  --run-id lewm_tworoom_full_seed0_n64 \
  --env tworoom \
  --model-family lewm_hf \
  --output-root outputs/result_bundles

uv run python scripts/concat_result_bundles.py \
  --bundle-roots \
    outputs/result_bundles/lewm_tworoom_full_seed0_n64 \
    outputs/result_bundles/lewm_pusht_full_seed0_n64 \
    outputs/result_bundles/swm_prejepa_tworoom_full_seed0_n64 \
  --output outputs/tables/all_frontiers_combined.csv
```
