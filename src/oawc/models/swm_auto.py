from __future__ import annotations

import torch
import stable_worldmodel as swm

from oawc.models.registry import LoadedCostModel, register_model_loader


@register_model_loader("lewm_hf")
def load_lewm_hf(
    *,
    checkpoint: str,
    env_name: str,
    device: str | torch.device,
) -> LoadedCostModel:
    """
    Load official LeWM HuggingFace checkpoints.

    Important: these checkpoints use config targets from the LeWM/SWM source tree
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
