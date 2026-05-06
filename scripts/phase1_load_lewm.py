from __future__ import annotations

import argparse

import torch

from oawc.config import get_run_config
from oawc.models.lewm_loader import count_parameters, load_lewm_from_hf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = get_run_config(args.run)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = load_lewm_from_hf(cfg.checkpoint_repo, device=device)

    print(f"Loaded LeWM checkpoint: {cfg.checkpoint_repo}")
    print(f"Device: {device}")
    print(f"Model class: {type(model).__name__}")
    print(f"Parameter count: {count_parameters(model):,}")
    print(f"Training mode: {model.training}")


if __name__ == "__main__":
    main()