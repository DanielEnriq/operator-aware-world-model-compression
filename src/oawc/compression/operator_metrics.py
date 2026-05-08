from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from oawc.benchmark import (
    fit_world_policy_processors,
    image_transform,
    load_hdf5_dataset,
)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. "
            "Use --device cpu locally or run on Colab/H100."
        )
    return device_arg


def extract_init_goal(dataset, episodes_idx, start_steps, goal_offset):
    ep_idx_arr = np.array(episodes_idx)
    start_arr = np.array(start_steps)
    data = dataset.load_chunk(
        ep_idx_arr,
        start_arr,
        start_arr + goal_offset + 1,
    )

    init_lists: dict[str, list] = {}
    goal_lists: dict[str, list] = {}

    for ep in data:
        for col in dataset.column_names:
            if col.startswith("goal"):
                continue
            val = ep[col]
            if isinstance(val, torch.Tensor):
                arr = val.detach().cpu().numpy()
            elif isinstance(val, np.ndarray):
                arr = val
            else:
                continue

            if col == "pixels" and arr.ndim >= 4:
                arr = np.transpose(arr, (0, 2, 3, 1))

            init_lists.setdefault(col, []).append(arr[0])
            goal_lists.setdefault(col, []).append(arr[-1])

    init_state = {k: np.stack(v) for k, v in init_lists.items()}
    goal_state = {}
    for k, v in goal_lists.items():
        goal_state["goal" if k == "pixels" else f"goal_{k}"] = np.stack(v)

    return init_state, goal_state


def build_info_dict_from_cache(
    *,
    env_name: str,
    episodes_idx: list[int],
    start_steps: list[int],
    goal_offset_steps: int,
    device: str,
) -> dict[str, torch.Tensor]:
    dataset = load_hdf5_dataset(env_name)
    init_state, goal_state = extract_init_goal(
        dataset,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset=goal_offset_steps,
    )

    info_np: dict[str, Any] = {}
    info_np.update(init_state)
    info_np.update(goal_state)
    info_np["id"] = np.asarray(episodes_idx, dtype=np.int64)
    info_np["step_idx"] = np.asarray(start_steps, dtype=np.int64)

    process = fit_world_policy_processors(
        dataset,
        keys_to_process=["action", "proprio"],
    )
    for key, scaler in process.items():
        if key not in info_np:
            continue
        arr = info_np[key]
        arr_2d = arr[:, None] if arr.ndim == 1 else arr.reshape(arr.shape[0], -1)
        info_np[key] = scaler.transform(arr_2d)

    img_tf = image_transform(224)

    def transform_images(arr: np.ndarray) -> torch.Tensor:
        out = [img_tf(img) for img in arr]
        return torch.stack(out, dim=0).unsqueeze(1)

    if "pixels" in info_np:
        info_np["pixels"] = transform_images(info_np["pixels"])
    if "goal" in info_np:
        info_np["goal"] = transform_images(info_np["goal"])

    info_torch: dict[str, torch.Tensor] = {}
    for key, value in info_np.items():
        if torch.is_tensor(value):
            tensor = value
        elif key in ("id", "step_idx"):
            tensor = torch.as_tensor(value, dtype=torch.int64).unsqueeze(1)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32).unsqueeze(1)
        info_torch[key] = tensor.to(device)
    return info_torch


def get_model_action_width(model: torch.nn.Module) -> int | None:
    action_encoder = getattr(model, "action_encoder", None)
    patch_embed = getattr(action_encoder, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "in_channels"):
        return int(patch_embed.in_channels)
    extra_encoders = getattr(model, "extra_encoders", None)
    if extra_encoders is not None and "action" in extra_encoders:
        patch_embed = getattr(extra_encoders["action"], "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "in_channels"):
            return int(patch_embed.in_channels)
    return None


def maybe_align_action_width(
    info_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    action = info_dict.get("action")
    if action is None:
        return info_dict
    expected = get_model_action_width(model)
    if expected is None:
        return info_dict
    current = int(action.shape[-1])
    if current == expected:
        return info_dict
    if expected % current != 0:
        raise ValueError(f"Action width mismatch: expected {expected}, got {current}")
    info_dict["action"] = action.repeat_interleave(expected // current, dim=-1)
    return info_dict


def adapt_candidates_for_model(
    candidate_actions: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    expected = get_model_action_width(model)
    if expected is None:
        return candidate_actions
    current = int(candidate_actions.shape[-1])
    if current == expected:
        return candidate_actions
    if expected % current != 0:
        raise ValueError(
            f"Candidate action width mismatch: expected {expected}, got {current}"
        )
    return candidate_actions.repeat_interleave(expected // current, dim=-1)


def expand_info_for_candidates(
    info_dict: dict[str, torch.Tensor],
    num_candidates: int,
) -> dict[str, torch.Tensor]:
    expanded: dict[str, torch.Tensor] = {}
    for key, value in info_dict.items():
        repeats = [1, num_candidates] + [1] * (value.ndim - 1)
        expanded[key] = value.unsqueeze(1).repeat(*repeats)
    return expanded


def _compute_cost_with_jepa_fallback(
    model: torch.nn.Module,
    info_dict: dict[str, torch.Tensor],
    action_candidates: torch.Tensor,
) -> torch.Tensor:
    # Operate directly on caller-provided chunked tensors to avoid cloning
    # large expanded inputs (e.g. pixels) during fallback.
    goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
    goal["pixels"] = goal["goal"]
    for key in list(goal.keys()):
        if key.startswith("goal_"):
            goal[key[len("goal_") :]] = goal.pop(key)
    goal.pop("action", None)
    goal = model.encode(goal)
    goal_emb = goal["emb"]

    rollout_input = dict(info_dict)
    rollout_input["goal_emb"] = goal_emb
    rollout_output = model.rollout(rollout_input, action_candidates)
    pred_emb = rollout_output["predicted_emb"]

    while goal_emb.ndim < pred_emb.ndim:
        goal_emb = goal_emb.unsqueeze(1)
    goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
    return F.mse_loss(
        pred_emb[..., -1:, :],
        goal_emb[..., -1:, :].detach(),
        reduction="none",
    ).sum(dim=tuple(range(2, pred_emb.ndim)))


def compute_model_costs(
    model: torch.nn.Module,
    info_dict: dict[str, torch.Tensor],
    action_candidates: torch.Tensor,
) -> torch.Tensor:
    try:
        return model.get_cost(info_dict, action_candidates)
    except RuntimeError as exc:
        msg = str(exc)
        if (
            "expanded size of the tensor" in msg
            and hasattr(model, "rollout")
            and hasattr(model, "encode")
        ):
            return _compute_cost_with_jepa_fallback(
                model,
                info_dict,
                action_candidates,
            )
        raise


def load_model_from_path(model_path: str | Path, device: str) -> torch.nn.Module:
    project_root = Path(__file__).resolve().parents[3]
    lewm_src = project_root / "external" / "le-wm"
    if lewm_src.exists() and str(lewm_src) not in sys.path:
        sys.path.insert(0, str(lewm_src))

    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    model = getattr(loaded, "model", loaded)
    if not hasattr(model, "get_cost"):
        raise TypeError(f"Loaded object from {model_path} does not expose get_cost().")
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model
