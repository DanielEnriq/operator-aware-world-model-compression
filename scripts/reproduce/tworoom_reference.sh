#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$PWD/src}"
export STABLEWM_HOME="${STABLEWM_HOME:-$PWD/.swm_cache}"

echo "PYTHONPATH=$PYTHONPATH"
echo "STABLEWM_HOME=$STABLEWM_HOME"

# 1. Acquire official LeWM TwoRoom dataset.
uv run python scripts/fetch_lewm_datasets.py --datasets tworoom

# 2. Validate dataset + environment + official eval task sampling.
uv run python scripts/check_benchmark_data.py \
  --env tworoom \
  --num-eval 8 \
  --seed 0

# 3. Random policy reference.
uv run python scripts/benchmark_random_baseline.py \
  --env tworoom \
  --num-eval 50 \
  --seed 0

# 4. Original LeWM reference.
DEVICE="${DEVICE:-auto}"
uv run python scripts/benchmark_cost_model.py \
  --env tworoom \
  --model-family lewm_hf \
  --checkpoint quentinll/lewm-tworooms \
  --tag lewm_tworoom_original \
  --num-eval 50 \
  --seed 0 \
  --device "$DEVICE"
