# Checkpoint Artifact Layout

This project keeps *reproducibility metadata* in git, but does **not** use git as the primary transport for large model artifacts.

## Layout

```text
checkpoints/
  swm_prejepa/
    tworoom/<run_id>/
    pusht/<run_id>/
    ogbench_cube/<run_id>/
  swm_pldm/
    tworoom/<run_id>/
    pusht/<run_id>/
    ogbench_cube/<run_id>/
```

Each run directory should include:

- `metadata.json` (required): provenance, command, env/family, artifact locations
- `config.yaml` or `config.json` (recommended): resolved training config
- Optional large artifacts (ignored by default): `weights_epoch_*.pt`, `*_object.ckpt`, logs

## Transport and Storage Policy

- Git is **not** the checkpoint transport layer for this repository.
- In Colab, train locally first, then copy final checkpoints to Google Drive.
- For local benchmarking, either:
  - copy run folders into `checkpoints/...`, or
  - pass an explicit checkpoint/run path to benchmark scripts.

## Official vs Reproduced Models

- Official LeWM checkpoints are downloaded from Hugging Face (e.g. `quentinll/lewm-*`).
- SWM PreJEPA and PLDM checkpoints trained from this repo are **our reproductions** and should be tagged/documented accordingly.
