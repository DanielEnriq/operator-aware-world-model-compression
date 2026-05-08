from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from oawc.benchmark import (
    detect_episode_column,
    get_episode_lengths,
    image_transform,
)


def sample_valid_row_indices(
    dataset,
    *,
    horizon: int,
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ep_col = detect_episode_column(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(ep_col), return_index=True)
    episode_len = get_episode_lengths(dataset, ep_indices)
    max_start_idx = episode_len - horizon
    max_start_idx_dict = {
        ep_id: max_start_idx[i]
        for i, ep_id in enumerate(ep_indices)
    }

    episode_per_row = dataset.get_col_data(ep_col)
    step_per_row = dataset.get_col_data("step_idx")
    max_start_per_row = np.asarray(
        [max_start_idx_dict[ep_id] for ep_id in episode_per_row]
    )
    valid_mask = step_per_row <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) == 0:
        raise ValueError("No valid rows found for requested horizon.")

    chosen = rng.choice(valid_indices, size=batch_size, replace=True)
    return np.asarray(chosen, dtype=np.int64)


def _extract_action_sequence(ep, horizon: int) -> np.ndarray:
    action = ep["action"]
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    action = np.asarray(action)
    if action.shape[0] >= horizon:
        return action[:horizon]

    # Right-pad with last action if short (rare edge case).
    pad = np.repeat(action[-1:][None, ...], horizon - action.shape[0], axis=0)
    pad = pad.reshape(horizon - action.shape[0], action.shape[1])
    return np.concatenate([action, pad], axis=0)


def build_prediction_batch(
    dataset,
    *,
    row_indices: np.ndarray,
    horizon: int,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    # h5py fancy indexing expects increasing order.
    row_indices = np.sort(np.asarray(row_indices, dtype=np.int64))
    rows = dataset.get_row_data(row_indices)
    ep_col = detect_episode_column(dataset)
    episodes = rows[ep_col].astype(int)
    starts = rows["step_idx"].astype(int)
    data = dataset.load_chunk(episodes, starts, starts + horizon)

    tf = image_transform(224)
    pixels_list = []
    actions_list = []

    for ep in data:
        pixels = ep["pixels"]
        if torch.is_tensor(pixels):
            px0 = pixels[0].detach().cpu().numpy()
        else:
            px0 = np.asarray(pixels[0])
        # CHW -> HWC expected by torchvision image pipeline in this project.
        if px0.ndim == 3 and px0.shape[0] in (1, 3):
            px0 = np.transpose(px0, (1, 2, 0))
        pixels_list.append(tf(px0))
        actions_list.append(_extract_action_sequence(ep, horizon))

    pixels_t = (
        torch.stack(pixels_list, dim=0)
        .unsqueeze(1)
        .unsqueeze(1)
        .to(device)
    )
    actions_t = torch.as_tensor(
        np.stack(actions_list),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    info = {"pixels": pixels_t}
    return info, actions_t


def clone_info_dict(info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in info.items()
    }


def predictor_param_partition(
    model: torch.nn.Module,
    trainable_substring: str,
) -> tuple[int, int]:
    trainable = 0
    frozen = 0
    for name, p in model.named_parameters():
        if trainable_substring in name:
            trainable += p.numel()
        else:
            frozen += p.numel()
    return int(trainable), int(frozen)


def set_trainable_by_substring(
    model: torch.nn.Module,
    substring: str,
) -> list[torch.nn.Parameter]:
    trainable = []
    for name, p in model.named_parameters():
        if substring in name:
            p.requires_grad = True
            trainable.append(p)
        else:
            p.requires_grad = False
    return trainable


def load_inherited_compression_report(
    student_path: str | Path,
) -> dict | None:
    path = Path(student_path)
    report = path.parent / "compression_report.json"
    if not report.exists():
        return None
    import json

    data = json.loads(report.read_text())
    return {
        "base_method": data.get("method"),
        "rank_fraction": data.get("rank_fraction"),
        "predictor_compression_ratio": data.get("predictor_compression_ratio"),
        "total_compression_ratio": data.get("total_compression_ratio"),
    }
