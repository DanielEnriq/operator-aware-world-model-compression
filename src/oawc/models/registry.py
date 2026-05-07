from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class LoadedCostModel:
    model: torch.nn.Module
    family: str
    checkpoint: str
    source: str


_MODEL_LOADERS: dict[str, Callable[..., LoadedCostModel]] = {}


def register_model_loader(family: str):
    def decorator(fn: Callable[..., LoadedCostModel]):
        if family in _MODEL_LOADERS:
            raise ValueError(f"Duplicate model loader for family={family}")
        _MODEL_LOADERS[family] = fn
        return fn

    return decorator


def available_model_families() -> list[str]:
    return sorted(_MODEL_LOADERS)


def load_cost_model(
    *,
    family: str,
    checkpoint: str,
    env_name: str,
    device: str | torch.device,
) -> LoadedCostModel:
    if family not in _MODEL_LOADERS:
        valid = ", ".join(available_model_families())
        raise KeyError(f"Unknown model family '{family}'. Valid families: {valid}")

    loaded = _MODEL_LOADERS[family](
        checkpoint=checkpoint,
        env_name=env_name,
        device=device,
    )

    model = loaded.model.to(device)
    model.eval()
    model.requires_grad_(False)

    # Several SWM/ViT-style cost models expose this inference-time flag.
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    return LoadedCostModel(
        model=model,
        family=loaded.family,
        checkpoint=loaded.checkpoint,
        source=loaded.source,
    )
