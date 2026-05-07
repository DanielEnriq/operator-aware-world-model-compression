from __future__ import annotations

import argparse

import torch

from oawc.config import get_run_config
from oawc.models.lewm_loader import count_parameters, load_lewm_from_hf
from oawc.models.lewm_wrapper import LeWMWrapper


def count_module_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = get_run_config(args.run)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    model = load_lewm_from_hf(cfg.checkpoint_repo, device=device)
    wrapper = LeWMWrapper(model=model, device=device, history_size=cfg.history_size)

    print(f"Loaded wrapped LeWM: {cfg.checkpoint_repo}")
    print(f"Device: {device}")
    print(f"Total parameters: {count_parameters(model):,}")

    print("\nCompression targets:")
    for name, module in wrapper.compression_targets().items():
        print(f"  {name}: {count_module_params(module):,} parameters")


if __name__ == "__main__":
    main()