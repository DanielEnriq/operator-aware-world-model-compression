# Colab Training Workflow (SWM PreJEPA / PLDM)

For large runs, prefer writing checkpoints to local Colab storage during training and copying final artifacts to Drive at the end.

## Important lane separation

- Official LeWM benchmark lane (installed SWM):
  - `PYTHONPATH="$PWD/src"`
  - Use `scripts/benchmark_cost_model.py` for official LeWM comparisons.
- Source-SWM lane (training + source checkpoint benchmarking):
  - `PYTHONPATH="$PWD/external/stable-worldmodel:$PWD/src"`
  - Use `scripts/train_world_model.py` and `scripts/benchmark_source_swm_checkpoint.py`.
- `scripts/benchmark_source_swm_checkpoint.py` is only for source-trained SWM checkpoints.
- On CPU, source checkpoint benchmark must use `--smoke-planner`.
- Full CEM source checkpoint benchmark should run on GPU/H100.

## 1) Clone repo

```bash
!git clone https://github.com/DanielEnriq/operator-aware-world-model-compression.git
%cd operator-aware-world-model-compression
```

## 2) Install uv + dependencies

```bash
!pip install uv
!uv sync
```

## 3) Clone upstream SWM source

```bash
!git clone https://github.com/galilai-group/stable-worldmodel external/stable-worldmodel
```

## 4) Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 5) Set environment

```python
import os
os.environ["PYTHONPATH"] = f"{os.getcwd()}/external/stable-worldmodel:{os.getcwd()}/src"
os.environ["STABLEWM_HOME"] = f"{os.getcwd()}/.swm_cache"
```

## 6) Download datasets

```bash
!uv run python scripts/fetch_lewm_datasets.py --datasets tworoom
# Optionally include pusht and ogbench_cube as needed.
```

## 7) Check training setup

```bash
!uv run python scripts/check_swm_training_setup.py
```

## 8) Smoke train

```bash
!uv run python scripts/train_world_model.py \
  --family swm_prejepa \
  --env tworoom \
  --mode smoke \
  --device cuda \
  --drive-output-root /content/drive/MyDrive/oawc_checkpoints \
  --copy-artifacts
```

## 9) Full train

```bash
!uv run python scripts/train_world_model.py \
  --family swm_prejepa \
  --env tworoom \
  --mode full \
  --device cuda \
  --drive-output-root /content/drive/MyDrive/oawc_checkpoints \
  --copy-artifacts
```

## 10) Inspect trained checkpoints

```bash
!uv run python scripts/inspect_trained_checkpoints.py --root /content/drive/MyDrive/oawc_checkpoints
```

## 11) Benchmark

```bash
!uv run python scripts/benchmark_cost_model.py \
  --env tworoom \
  --model-family swm_auto \
  --checkpoint <run_name> \
  --tag <tag> \
  --num-eval 50 \
  --seed 0 \
  --device cuda
```

## 12) Benchmark source-trained SWM checkpoint (isolated lane)

```bash
!uv run python scripts/benchmark_source_swm_checkpoint.py \
  --env tworoom \
  --checkpoint <run_name> \
  --tag <tag> \
  --num-eval 1 \
  --seed 0 \
  --device cpu \
  --smoke-planner
```
