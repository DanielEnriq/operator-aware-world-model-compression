from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from oawc.config import RunConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def phase1_run_dir(cfg: RunConfig) -> Path:
    return OUTPUT_ROOT / "phase1" / cfg.run_name


def phase1_dirs(cfg: RunConfig) -> dict[str, Path]:
    root = phase1_run_dir(cfg)

    dirs = {
        "run": root,
        "data": root / "data",
        "latents": root / "latents",
        "windows": root / "windows",
        "results": root / "results",
        "figures": root / "figures",
        "videos": root / "videos",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(x):
        if isinstance(x, Path):
            return str(x)
        if hasattr(x, "item"):
            return x.item()
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        return x

    with path.open("w") as f:
        json.dump(convert(payload), f, indent=2)


def save_config(path: Path, cfg: RunConfig) -> None:
    save_json(path, asdict(cfg))