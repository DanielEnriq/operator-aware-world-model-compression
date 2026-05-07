from __future__ import annotations

import argparse

import torch

from oawc.config import get_run_config
from oawc.models.lewm_loader import count_parameters, load_lewm_from_hf
from oawc.models.lewm_wrapper import LeWMWrapper


def count_module_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def pct(part: int, total: int) -> float:
    return 100.0 * part / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = get_run_config(args.run)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    model = load_lewm_from_hf(cfg.checkpoint_repo, device=device)
    wrapper = LeWMWrapper(model=model, device=device, history_size=cfg.history_size)

    total = count_parameters(model)

    observation_modules = {
        "encoder": model.encoder,
        "projector": model.projector,
    }

    transition_modules = wrapper.compression_targets()

    observation_total = sum(count_module_params(m) for m in observation_modules.values())
    transition_total = sum(count_module_params(m) for m in transition_modules.values())

    accounted_total = observation_total + transition_total
    unaccounted = total - accounted_total

    print(f"Loaded wrapped LeWM: {cfg.checkpoint_repo}")
    print(f"Device: {device}")
    print(f"Total parameters: {total:,}")

    print("\nObservation / representation modules:")
    for name, module in observation_modules.items():
        n = count_module_params(module)
        print(f"  {name:<16} {n:>12,} params  ({pct(n, total):5.1f}%)")

    print("\nTransition / planning modules:")
    for name, module in transition_modules.items():
        n = count_module_params(module)
        print(f"  {name:<16} {n:>12,} params  ({pct(n, total):5.1f}%)")

    print("\nSummary:")
    print(f"  observation_total {observation_total:>12,} params  ({pct(observation_total, total):5.1f}%)")
    print(f"  transition_total  {transition_total:>12,} params  ({pct(transition_total, total):5.1f}%)")
    print(f"  accounted_total   {accounted_total:>12,} params  ({pct(accounted_total, total):5.1f}%)")
    print(f"  unaccounted       {unaccounted:>12,} params  ({pct(unaccounted, total):5.1f}%)")

    print("\nCompression priority:")
    print("  1. predictor only")
    print("  2. predictor + pred_proj")
    print("  3. action_encoder + predictor + pred_proj")
    print("  4. full model: encoder + projector + transition modules")


if __name__ == "__main__":
    main()
