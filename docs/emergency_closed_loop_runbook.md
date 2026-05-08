# Emergency Closed-Loop Runbook (Multi-Env)

This runbook reports only direct closed-loop MPC benchmark results from
`scripts/benchmark_cost_model.py`.

It does not use random-cache/operator-cache/crossed-split pipelines.

## Cell 1: Fresh clone, setup, fetch all datasets

```bash
!git clone https://github.com/DanielEnriq/operator-aware-world-model-compression.git
%cd operator-aware-world-model-compression

!pip install -q uv
!uv sync
!git clone https://github.com/lucas-maes/le-wm.git external/le-wm

import os
os.environ["PYTHONPATH"] = f"{os.getcwd()}/src"
os.environ["STABLEWM_HOME"] = f"{os.getcwd()}/.swm_cache"

!uv run python scripts/fetch_lewm_datasets.py --datasets tworoom pusht ogbench_cube
```

## Cell 2: Teacher smoke for all environments

```bash
!uv run python scripts/benchmark_cost_model.py \
  --env tworoom \
  --model-family lewm_hf \
  --checkpoint quentinll/lewm-tworooms \
  --tag lewm_tworoom_teacher_smoke_seed0_n4 \
  --num-eval 4 \
  --seed 0 \
  --device cuda

!uv run python scripts/benchmark_cost_model.py \
  --env pusht \
  --model-family lewm_hf \
  --checkpoint quentinll/lewm-pusht \
  --tag lewm_pusht_teacher_smoke_seed0_n2 \
  --num-eval 2 \
  --seed 0 \
  --device cuda

!uv run python scripts/benchmark_cost_model.py \
  --env ogbench_cube \
  --model-family lewm_hf \
  --checkpoint quentinll/lewm-cube \
  --tag lewm_ogbench_cube_teacher_smoke_seed0_n2 \
  --num-eval 2 \
  --seed 0 \
  --device cuda
```

## Cell 3: Emergency multi-env frontier

```bash
!uv run python scripts/run_emergency_closed_loop_frontier.py \
  --envs tworoom,pusht,ogbench_cube \
  --device cuda \
  --ranks "0.95,0.90,0.80" \
  --methods "weight_svd" \
  --num-eval 8 \
  --seed 0 \
  --compress-if-missing \
  --skip-existing
```

## Cell 4: Print combined table

```bash
!cat outputs/tables/emergency_closed_loop_all.md
```
