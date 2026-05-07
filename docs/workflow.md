# Canonical Workflow

1. Install dependencies.
   - `uv sync`
2. Set source-first environment.
   - `export PYTHONPATH="$PWD/external/stable-worldmodel:$PWD/src"`
   - `export STABLEWM_HOME="$PWD/.swm_cache"`
3. Fetch datasets.
   - `uv run python scripts/fetch_lewm_datasets.py --datasets tworoom pusht ogbench_cube`
4. Validate benchmark data.
   - `uv run python scripts/check_benchmark_data.py --env tworoom --num-eval 4 --seed 0`
5. Run official LeWM benchmark (HF checkpoints).
   - `uv run python scripts/benchmark_cost_model.py --env tworoom --model-family lewm_hf --checkpoint quentinll/lewm-tworooms --tag lewm_tworoom --num-eval 50 --seed 0 --device cuda`
6. Train SWM PreJEPA/PLDM reproductions.
   - `uv run python scripts/train_world_model.py --family swm_prejepa --env tworoom --mode full --device cuda`
   - `uv run python scripts/train_world_model.py --family swm_pldm --env tworoom --mode full --device cuda`
7. Benchmark trained model.
   - `uv run python scripts/benchmark_cost_model.py --env tworoom --model-family swm_auto --checkpoint oawc_swm_prejepa_tworoom_seed0 --tag swm_prejepa_tworoom_seed0 --num-eval 50 --seed 0 --device cuda`
8. Compress model (project-specific compression path).
9. Benchmark compressed model.
10. Summarize results.
   - `uv run python scripts/summarize_benchmarks.py`
