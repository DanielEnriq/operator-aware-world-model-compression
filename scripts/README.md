# Scripts Inventory

| Script | Category | Status | Purpose | Typical command |
|---|---|---|---|---|
| `scripts/fetch_lewm_datasets.py` | data | canonical | Download official LeWM datasets into `$STABLEWM_HOME`. | `uv run python scripts/fetch_lewm_datasets.py --datasets tworoom pusht ogbench_cube` |
| `scripts/check_benchmark_data.py` | benchmark | canonical | Validate dataset/world availability and eval sampling. | `uv run python scripts/check_benchmark_data.py --env tworoom --num-eval 4 --seed 0` |
| `scripts/benchmark_cost_model.py` | benchmark | canonical | Dataset-driven control benchmark for cost models. | `uv run python scripts/benchmark_cost_model.py --env tworoom --model-family lewm_hf --checkpoint quentinll/lewm-tworooms --tag lewm_tworoom --num-eval 50 --seed 0 --device cuda` |
| `scripts/benchmark_source_swm_checkpoint.py` | benchmark | canonical (isolated source lane) | Benchmark source-trained SWM checkpoints using source SWM APIs. Not for official LeWM comparison benchmarks. | `uv run python scripts/benchmark_source_swm_checkpoint.py --env tworoom --checkpoint oawc_swm_prejepa_tworoom_seed0_smoke --tag prejepa_tworoom_smoke --num-eval 1 --seed 0 --device cpu --smoke-planner` |
| `scripts/build_operator_cache.py` | operator-aware | canonical | Build fixed-candidate teacher operator cache (`get_cost`) for compression research. | `uv run python scripts/build_operator_cache.py --env tworoom --model-family lewm_hf --checkpoint quentinll/lewm-tworooms --num-states 4 --num-candidates 16 --horizon 5 --topk 1 5 10 --seed 0 --device cpu --tag lewm_tworoom_smoke` |
| `scripts/check_operator_cache.py` | operator-aware | canonical | Validate operator cache tensor shapes, finiteness, and ranking consistency. | `uv run python scripts/check_operator_cache.py outputs/operator_cache/tworoom/lewm_tworoom_smoke/operator_cache.pt` |
| `scripts/benchmark_random_baseline.py` | benchmark | canonical | Random-policy baseline for context and sanity checks. | `uv run python scripts/benchmark_random_baseline.py --env tworoom --num-eval 50 --seed 0` |
| `scripts/summarize_benchmarks.py` | benchmark | canonical | Aggregate benchmark JSON outputs into summary views. | `uv run python scripts/summarize_benchmarks.py` |
| `scripts/check_model_artifacts.py` | inspect | active | Check known model artifact locations and optional loads. | `uv run python scripts/check_model_artifacts.py --try-load` |
| `scripts/check_swm_training_setup.py` | train | canonical | Validate SWM source import path, configs, scripts, and datasets. | `uv run python scripts/check_swm_training_setup.py` |
| `scripts/train_world_model.py` | train | canonical | Unified launcher for SWM PreJEPA/PLDM training orchestration. | `uv run python scripts/train_world_model.py --family swm_prejepa --env tworoom --mode smoke --device cpu --dry-run` |
| `scripts/inspect_trained_checkpoints.py` | inspect | canonical | Scan and load-check trained checkpoints, write JSON inspection report. | `uv run python scripts/inspect_trained_checkpoints.py` |
| `scripts/swm_inventory.py` | inspect | active | Local inventory helper for SWM-related assets. | `uv run python scripts/swm_inventory.py` |
| `scripts/phase1_*` | legacy | legacy / research scratch | Early phase-1 exploratory scripts for data/model probing. | Varies by script |
| `scripts/phase2_*` | legacy | legacy / research scratch | Early phase-2 compression/eval experiments. | Varies by script |
| `scripts/phase3_*` | legacy | legacy / research scratch | Early phase-3 distillation experiments. | Varies by script |

## Lane separation quick reference

- Official LeWM benchmark lane:
  - `export PYTHONPATH="$PWD/src"`
  - `export STABLEWM_HOME="$PWD/.swm_cache"`
  - `uv run python scripts/benchmark_cost_model.py ...`
- Source-SWM training and source checkpoint benchmark lane:
  - `export PYTHONPATH="$PWD/external/stable-worldmodel:$PWD/src"`
  - `export STABLEWM_HOME="$PWD/.swm_cache"`
  - `uv run python scripts/train_world_model.py --family swm_prejepa ...`
  - `uv run python scripts/benchmark_source_swm_checkpoint.py ... --smoke-planner`
- CPU source benchmark must use `--smoke-planner`; run full CEM on GPU/H100.

## Operator cache smoke commands

- Installed-SWM LeWM cache:
  - `export PYTHONPATH="$PWD/src"`
  - `export STABLEWM_HOME="$PWD/.swm_cache"`
  - `uv run python scripts/build_operator_cache.py --env tworoom --model-family lewm_hf --checkpoint quentinll/lewm-tworooms --num-states 4 --num-candidates 16 --horizon 5 --topk 1 5 10 --seed 0 --device cpu --tag lewm_tworoom_smoke`
  - `uv run python scripts/check_operator_cache.py outputs/operator_cache/tworoom/lewm_tworoom_smoke/operator_cache.pt`
- Source-SWM PreJEPA cache:
  - `export PYTHONPATH="$PWD/external/stable-worldmodel:$PWD/src"`
  - `export STABLEWM_HOME="$PWD/.swm_cache"`
  - `uv run python scripts/build_operator_cache.py --env tworoom --model-family swm_auto --checkpoint oawc_swm_prejepa_tworoom_seed0_smoke --source-swm --num-states 4 --num-candidates 16 --horizon 5 --topk 1 5 10 --seed 0 --device cpu --tag prejepa_tworoom_smoke`
  - `uv run python scripts/check_operator_cache.py outputs/operator_cache/tworoom/prejepa_tworoom_smoke/operator_cache.pt`

## Method 3 status note

- `scripts/distill_prediction_only.py` implements the prediction-only distillation baseline.
- Current LeWM TwoRoom safe run (`lewm_tworoom_svd_r050_pred_distill_safe`) fails operator-finiteness validation, so no deployable distilled model is saved.
- Until a run passes finite operator validation, Method 3 is tracked as unstable and not used as a competitive baseline.
