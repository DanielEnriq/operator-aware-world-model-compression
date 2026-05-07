# Scripts Inventory

| Script | Category | Status | Purpose | Typical command |
|---|---|---|---|---|
| `scripts/fetch_lewm_datasets.py` | data | canonical | Download official LeWM datasets into `$STABLEWM_HOME`. | `uv run python scripts/fetch_lewm_datasets.py --datasets tworoom pusht ogbench_cube` |
| `scripts/check_benchmark_data.py` | benchmark | canonical | Validate dataset/world availability and eval sampling. | `uv run python scripts/check_benchmark_data.py --env tworoom --num-eval 4 --seed 0` |
| `scripts/benchmark_cost_model.py` | benchmark | canonical | Dataset-driven control benchmark for cost models. | `uv run python scripts/benchmark_cost_model.py --env tworoom --model-family lewm_hf --checkpoint quentinll/lewm-tworooms --tag lewm_tworoom --num-eval 50 --seed 0 --device cuda` |
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
