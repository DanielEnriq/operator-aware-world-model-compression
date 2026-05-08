# Canonical Workflow

## Execution lanes (must stay isolated)

- Official LeWM benchmark lane (installed SWM):
  - `export PYTHONPATH="$PWD/src"`
  - `export STABLEWM_HOME="$PWD/.swm_cache"`
  - `uv run python scripts/benchmark_cost_model.py ...`
- Source-SWM trained checkpoint lane:
  - `export PYTHONPATH="$PWD/external/stable-worldmodel:$PWD/src"`
  - `export STABLEWM_HOME="$PWD/.swm_cache"`
  - `uv run python scripts/train_world_model.py --family swm_prejepa ...`
  - `uv run python scripts/benchmark_source_swm_checkpoint.py ... --smoke-planner`

Notes:
- Use `scripts/benchmark_cost_model.py` for official LeWM comparisons.
- Use `scripts/benchmark_source_swm_checkpoint.py` only for source-trained SWM checkpoints.
- On CPU, source-SWM benchmark must use `--smoke-planner`.
- Full CEM source-SWM benchmark should be run on GPU/H100.

1. Install dependencies.
   - `uv sync`
2. Set source training environment.
   - `export PYTHONPATH="$PWD/external/stable-worldmodel:$PWD/src"`
   - `export STABLEWM_HOME="$PWD/.swm_cache"`
3. Fetch datasets.
   - `uv run python scripts/fetch_lewm_datasets.py --datasets tworoom pusht ogbench_cube`
4. Validate benchmark data.
   - `uv run python scripts/check_benchmark_data.py --env tworoom --num-eval 4 --seed 0`
5. Run official LeWM benchmark (installed SWM lane).
   - `export PYTHONPATH="$PWD/src"`
   - `uv run python scripts/benchmark_cost_model.py --env tworoom --model-family lewm_hf --checkpoint quentinll/lewm-tworooms --tag lewm_tworoom --num-eval 50 --seed 0 --device cuda`
6. Train SWM PreJEPA/PLDM reproductions.
   - `export PYTHONPATH="$PWD/external/stable-worldmodel:$PWD/src"`
   - `uv run python scripts/train_world_model.py --family swm_prejepa --env tworoom --mode full --device cuda`
   - `uv run python scripts/train_world_model.py --family swm_pldm --env tworoom --mode full --device cuda`
7. Benchmark source-trained model (source-SWM lane).
   - `uv run python scripts/benchmark_source_swm_checkpoint.py --env tworoom --checkpoint oawc_swm_prejepa_tworoom_seed0 --tag swm_prejepa_tworoom_seed0 --num-eval 50 --seed 0 --device cuda`
8. Build teacher operator cache (stage-1 operator-aware scaffold, no compression yet).
   - Installed-SWM LeWM cache:
     - `export PYTHONPATH="$PWD/src"`
     - `uv run python scripts/build_operator_cache.py --env tworoom --model-family lewm_hf --checkpoint quentinll/lewm-tworooms --num-states 4 --num-candidates 16 --horizon 5 --topk 1 5 10 --seed 0 --device cpu --tag lewm_tworoom_smoke`
     - `uv run python scripts/check_operator_cache.py outputs/operator_cache/tworoom/lewm_tworoom_smoke/operator_cache.pt`
   - Source-SWM PreJEPA cache:
     - `export PYTHONPATH="$PWD/external/stable-worldmodel:$PWD/src"`
     - `uv run python scripts/build_operator_cache.py --env tworoom --model-family swm_auto --checkpoint oawc_swm_prejepa_tworoom_seed0_smoke --source-swm --num-states 4 --num-candidates 16 --horizon 5 --topk 1 5 10 --seed 0 --device cpu --tag prejepa_tworoom_smoke`
     - `uv run python scripts/check_operator_cache.py outputs/operator_cache/tworoom/prejepa_tworoom_smoke/operator_cache.pt`
9. Compress model (project-specific compression path).
10. Benchmark compressed model.
11. Summarize results.
   - `uv run python scripts/summarize_benchmarks.py`
