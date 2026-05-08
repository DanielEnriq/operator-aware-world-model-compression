from __future__ import annotations

import sys
from pathlib import Path

import torch
import stable_worldmodel as swm

from oawc.models.registry import LoadedCostModel, register_model_loader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEWM_SRC = PROJECT_ROOT / "external" / "le-wm"


def _ensure_lewm_source_on_path() -> None:
    if LEWM_SRC.exists() and str(LEWM_SRC) not in sys.path:
        sys.path.insert(0, str(LEWM_SRC))


@register_model_loader("lewm_hf")
def load_lewm_hf(
    *,
    checkpoint: str,
    env_name: str,
    device: str | torch.device,
) -> LoadedCostModel:
    """
    Load official LeWM HuggingFace checkpoints.

    Important: these checkpoints use config targets from the
    LeWM/SWM source tree
    that may not exist in the installed stable_worldmodel wheel. Therefore this
    intentionally delegates to our dedicated LeWM loader instead of using
    Hydra directly or swm.policy.AutoCostModel.
    """
    from oawc.models.lewm_loader import load_lewm_from_hf

    model = load_lewm_from_hf(checkpoint, device=str(device))

    return LoadedCostModel(
        model=model,
        family="lewm_hf",
        checkpoint=checkpoint,
        source="oawc.models.lewm_loader.load_lewm_from_hf",
    )


@register_model_loader("swm_auto")
def load_swm_auto(
    *,
    checkpoint: str,
    env_name: str,
    device: str | torch.device,
) -> LoadedCostModel:
    """
    Load local SWM object checkpoints known to swm.policy.AutoCostModel.
    This is not for quentinll/lewm-* HuggingFace repos.
    """
    model = swm.policy.AutoCostModel(checkpoint)

    return LoadedCostModel(
        model=model,
        family="swm_auto",
        checkpoint=checkpoint,
        source="stable_worldmodel.policy.AutoCostModel",
    )


@register_model_loader("swm_pretrained")
def load_swm_pretrained(
    *,
    checkpoint: str,
    env_name: str,
    device: str | torch.device,
) -> LoadedCostModel:
    """
    Load SWM save_pretrained-style checkpoints via config.json + weights .pt.

    This loader expects `checkpoint` to be either:
    - a run directory, or
    - a specific .pt file, or
    - a run name resolvable under $STABLEWM_HOME/checkpoints.
    """
    model = swm.wm.utils.load_pretrained(checkpoint)

    return LoadedCostModel(
        model=model,
        family="swm_pretrained",
        checkpoint=checkpoint,
        source="stable_worldmodel.wm.utils.load_pretrained",
    )


@register_model_loader("torch_file")
def load_torch_file(
    *,
    checkpoint: str,
    env_name: str,
    device: str | torch.device,
) -> LoadedCostModel:
    del env_name, device
    _ensure_lewm_source_on_path()
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        model = loaded
        source = "torch.load(nn.Module)"
    else:
        model = getattr(loaded, "model", loaded)
        source = (
            "torch.load(container.model)"
            if hasattr(loaded, "model")
            else "torch.load(raw_object)"
        )
    return LoadedCostModel(
        model=model,
        family="torch_file",
        checkpoint=checkpoint,
        source=source,
    )
