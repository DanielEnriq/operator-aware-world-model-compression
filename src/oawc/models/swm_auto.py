from __future__ import annotations

import re
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


def _epoch_num(path: Path) -> int:
    match = re.search(r"weights_epoch_(\d+)\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _load_module_like(path: Path) -> tuple[torch.nn.Module | None, str]:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        return loaded, "torch.load(nn.Module)"
    model = getattr(loaded, "model", None)
    if isinstance(model, torch.nn.Module):
        return model, "torch.load(container.model)"
    return None, f"torch.load({type(loaded).__name__})"


def _resolve_swm_prejepa_candidates(checkpoint: str) -> list[Path]:
    path = Path(checkpoint)
    if path.is_file():
        return [path]

    candidates: list[Path] = []
    object_ckpts = sorted(path.glob("*_object.ckpt"))
    epoch_pts = sorted(
        path.glob("weights_epoch_*.pt"),
        key=_epoch_num,
        reverse=True,
    )
    weights_ckpts = sorted(path.glob("*_weights.ckpt"))
    other_ckpt_pt = sorted(
        p
        for p in path.glob("*")
        if p.is_file() and p.suffix in {".ckpt", ".pt"}
    )
    candidates.extend(object_ckpts)
    candidates.extend(epoch_pts)
    candidates.extend(weights_ckpts)
    for p in other_ckpt_pt:
        if p not in candidates:
            candidates.append(p)
    return candidates


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


@register_model_loader("swm_prejepa_local")
def load_swm_prejepa_local(
    *,
    checkpoint: str,
    env_name: str,
    device: str | torch.device,
) -> LoadedCostModel:
    del env_name, device
    _ensure_lewm_source_on_path()
    candidates = _resolve_swm_prejepa_candidates(checkpoint)
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint artifacts found for swm_prejepa_local at "
            f"{checkpoint}. Expected *_object.ckpt, weights_epoch_*.pt, "
            "or *_weights.ckpt."
        )

    attempted: list[str] = []
    for candidate in candidates:
        try:
            model, source = _load_module_like(candidate)
            if isinstance(model, torch.nn.Module):
                return LoadedCostModel(
                    model=model,
                    family="swm_prejepa_local",
                    checkpoint=str(candidate),
                    source=source,
                )
            attempted.append(
                f"{candidate.name}: loaded non-module object"
            )
        except Exception as exc:
            attempted.append(f"{candidate.name}: {exc}")

    path = Path(checkpoint)
    if path.is_dir():
        try:
            pretrained_model = swm.wm.utils.load_pretrained(str(path))
            return LoadedCostModel(
                model=pretrained_model,
                family="swm_prejepa_local",
                checkpoint=str(path),
                source="stable_worldmodel.wm.utils.load_pretrained",
            )
        except Exception as exc:
            attempted.append(f"load_pretrained({path}): {exc}")

    details = "; ".join(attempted[:12])
    raise ValueError(
        "Unable to load swm_prejepa_local as a full torch.nn.Module. "
        "Checked candidates in priority order "
        "(*_object.ckpt, weights_epoch_*.pt, *_weights.ckpt, others). "
        f"Attempts: {details}"
    )
