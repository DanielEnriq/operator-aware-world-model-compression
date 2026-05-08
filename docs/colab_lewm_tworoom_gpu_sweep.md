# Colab LeWM TwoRoom GPU Sweep

This workflow runs the installed-SWM LeWM compression/distillation sweep with
train/eval operator-cache split and saves outputs to Google Drive.

## Cell 1: Clone Repo

```bash
%%bash
cd /content
if [ ! -d operator-aware-world-model-compression ]; then
  git clone https://github.com/danielenriquez/operator-aware-world-model-compression.git
fi
cd /content/operator-aware-world-model-compression
git pull
```

## Cell 2: Install UV + Dependencies

```bash
%%bash
cd /content/operator-aware-world-model-compression
python -m pip install -U pip uv
uv sync
```

## Cell 3: Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## Cell 4: Set Installed-Lane Env Vars

```bash
%%bash
cd /content/operator-aware-world-model-compression
export PYTHONPATH="$PWD/src"
export STABLEWM_HOME="$PWD/.swm_cache"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED=1
echo "PYTHONPATH=$PYTHONPATH"
echo "STABLEWM_HOME=$STABLEWM_HOME"
```

## Cell 5: Verify GPU + Installed Stable WorldModel Lane

```bash
%%bash
cd /content/operator-aware-world-model-compression
export PYTHONPATH="$PWD/src"
export STABLEWM_HOME="$PWD/.swm_cache"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED=1
uv run python - <<'PY'
import torch, stable_worldmodel, sys
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("stable_worldmodel:", stable_worldmodel.__file__)
if "external/stable-worldmodel" in stable_worldmodel.__file__:
    raise SystemExit("ERROR: wrong lane, imported source SWM path")
print("lane check passed: installed SWM")
PY
```

## Cell 6: Fetch TwoRoom Dataset

```bash
%%bash
cd /content/operator-aware-world-model-compression
export PYTHONPATH="$PWD/src"
export STABLEWM_HOME="$PWD/.swm_cache"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED=1
uv run python scripts/check_benchmark_data.py --env tworoom --num-eval 4 --seed 0
```

## Cell 7: Run Dry-Run Sweep

```bash
%%bash
cd /content/operator-aware-world-model-compression
export PYTHONPATH="$PWD/src"
export STABLEWM_HOME="$PWD/.swm_cache"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED=1
uv run python scripts/run_lewm_tworoom_gpu_sweep.py \
  --drive-root /content/drive/MyDrive/oawc_lewm_tworoom_sweep \
  --device cuda \
  --env tworoom \
  --teacher-checkpoint quentinll/lewm-tworooms \
  --train-states 512 \
  --eval-states 128 \
  --num-candidates 128 \
  --horizon 5 \
  --seed-train 0 \
  --seed-eval 1 \
  --distill-steps 100 \
  --distill-batch-size 8 \
  --cost-batch-states 8 \
  --cost-batch-candidates 128 \
  --lr 1e-5 \
  --dry-run
```

## Cell 8: Run Full Sweep

```bash
%%bash
cd /content/operator-aware-world-model-compression
export PYTHONPATH="$PWD/src"
export STABLEWM_HOME="$PWD/.swm_cache"
export MPLBACKEND="Agg"
export PYTHONUNBUFFERED=1
uv run python scripts/run_lewm_tworoom_gpu_sweep.py \
  --drive-root /content/drive/MyDrive/oawc_lewm_tworoom_sweep \
  --device cuda \
  --env tworoom \
  --teacher-checkpoint quentinll/lewm-tworooms \
  --train-states 512 \
  --eval-states 128 \
  --num-candidates 128 \
  --horizon 5 \
  --seed-train 0 \
  --seed-eval 1 \
  --distill-steps 100 \
  --distill-batch-size 8 \
  --cost-batch-states 8 \
  --cost-batch-candidates 128 \
  --lr 1e-5 \
  --skip-existing
```

## Cell 9: Inspect Held-out Summary Table

```bash
%%bash
cd /content/operator-aware-world-model-compression
python - <<'PY'
import json, pathlib
csv_path = pathlib.Path("outputs/tables/operator_split_summary_tworoom.csv")
json_path = pathlib.Path("outputs/tables/final_benchmark_candidates.json")
print("summary csv exists:", csv_path.exists(), csv_path)
print("candidates json exists:", json_path.exists(), json_path)
if json_path.exists():
    data = json.loads(json_path.read_text())
    print("num candidates:", len(data.get("candidates", [])))
PY
```

## Cell 10: Check Drive Outputs

```bash
%%bash
ls -R /content/drive/MyDrive/oawc_lewm_tworoom_sweep | head -n 200
```

Notes:
- Installed lane only: `PYTHONPATH="$PWD/src"`.
- Do not set `PYTHONPATH="$PWD/external/stable-worldmodel:$PWD/src"` for this
  sweep.
- This workflow does not run full CEM environment benchmarks beyond regression
  guard.
