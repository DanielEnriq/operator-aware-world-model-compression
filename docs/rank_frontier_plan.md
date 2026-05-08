# TwoRoom Rank Frontier Plan

This plan defines the next research iteration for Operator-Aware World Model
Compression on LeWM TwoRoom, with explicit separation between:

- random-candidate operator stress tests,
- dataset-action operator fidelity,
- near-elite/planner-local operator fidelity,
- closed-loop MPC success.

## Stage 0: Sanity

```bash
uv run python scripts/check_benchmark_data.py --env tworoom --num-eval 4 --seed 0
uv run python scripts/benchmark_cost_model.py \
  --env tworoom \
  --model-family lewm_hf \
  --checkpoint quentinll/lewm-tworooms \
  --tag lewm_tworoom_teacher_regression \
  --num-eval 4 \
  --seed 0 \
  --device cpu
uv run python scripts/smoke_rank_frontier_setup.py --env tworoom --device cpu
```

## Stage 1: Baseline Rank Frontier (No Distillation)

```bash
uv run python scripts/run_lewm_tworoom_rank_frontier.py \
  --env tworoom \
  --teacher-checkpoint quentinll/lewm-tworooms \
  --device cuda \
  --rank-fractions "1.0,0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50" \
  --methods "weight_svd,aa_svd" \
  --eval-random-cache true \
  --eval-cache outputs/operator_cache/tworoom/lewm_tworoom_eval_s128_c128_seed1/operator_cache.pt \
  --train-cache outputs/operator_cache/tworoom/lewm_tworoom_train_s512_c128_seed0/operator_cache.pt \
  --skip-existing
```

## Stage 2: Dataset-Action Operator Cache

```bash
uv run python scripts/build_dataset_action_operator_cache.py \
  --env tworoom \
  --model-family lewm_hf \
  --checkpoint quentinll/lewm-tworooms \
  --num-states 128 \
  --num-candidates 128 \
  --horizon 5 \
  --seed 2 \
  --device cuda \
  --tag lewm_tworoom_eval_dataset_actions_s128_c128_seed2 \
  --candidate-mode dataset_actions \
  --action-noise-std 0.0
```

Evaluate selected compressed models on that cache using canonical evaluator:

```bash
uv run python scripts/evaluate_operator_metrics.py \
  --cache outputs/operator_cache/tworoom/lewm_tworoom_eval_dataset_actions_s128_c128_seed2/operator_cache.pt \
  --model-path outputs/compression/tworoom/lewm_tworoom_aa_svd_r095/compressed_model.pt \
  --device cuda \
  --tag lewm_tworoom_aa_svd_r095_eval_dataset_actions_s128_seed2
```

## Stage 3: Near-Elite Cache (Planner-Local Proxy)

```bash
uv run python scripts/build_near_elite_operator_cache.py \
  --source-cache outputs/operator_cache/tworoom/lewm_tworoom_eval_s128_c128_seed1/operator_cache.pt \
  --tag lewm_tworoom_eval_near_elite_s128_c128_seed3 \
  --elite-k 10 \
  --noise-std 0.05 \
  --seed 3 \
  --device cuda
```

Then evaluate selected compressed models with `scripts/evaluate_operator_metrics.py`.

## Stage 4: Closed-Loop Smoke

```bash
uv run python scripts/run_closed_loop_rank_frontier.py \
  --env tworoom \
  --device cuda \
  --num-eval 16 \
  --seed 0 \
  --ranks "0.95,0.90,0.80,0.70,0.60,0.50" \
  --methods "svd,aa_svd" \
  --skip-existing
```

## Stage 5: Distillation Follow-Up (Only Where Useful)

After frontier baselines exist, run operator-aware distillation for:

- ranks: `0.90, 0.80, 0.70, 0.60, 0.50`
- methods: cost-KL and hybrid
- validation defaults: `--val-states 64 --val-candidates 128 --eval-every 25`

## Summaries and Plots

```bash
uv run python scripts/summarize_rank_frontier.py --env tworoom
uv run python scripts/plot_rank_frontier.py \
  --summary-json outputs/tables/rank_frontier_tworoom.json \
  --output-dir outputs/figures/rank_frontier_tworoom
```

Scientific framing:

- `r=0.25` and `r=0.5` alone are insufficient to claim compression failure.
- Random action caches are stress tests, not final planning truth.
- Main question: where does behavior collapse along compression ratio?
- Equal-ratio and equal-compute comparisons are both required.
- Closed-loop MPC remains the final arbiter.
