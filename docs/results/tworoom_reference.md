# TwoRoom Reference Benchmark

Paper-faithful TwoRoom contract matched to `external/le-wm/config/eval/tworoom.yaml`.

## Protocol

- Environment: `swm/TwoRoom-v1`
- Dataset: `tworoom`
- World history size: `1`
- World frame skip: `1`
- Plan horizon: `5`
- Receding horizon: `5`
- Action block: `5`
- Goal offset: `25`
- Eval budget: `50`
- Evaluation: dataset-driven start/goal replay via `World.evaluate_from_dataset`

## Local CPU reference results

| Model | n | Seed | Success rate | Notes |
|---|---:|---:|---:|---|
| RandomPolicy | 50 | 0 | 4.0% | Same dataset-driven task selection |
| Original LeWM `quentinll/lewm-tworooms` | 50 | 0 | 88.0% | CEM, CPU |

Generated JSON files are stored under `outputs/benchmarks/`, which is intentionally git-ignored.
