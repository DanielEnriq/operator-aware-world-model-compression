from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from oawc.compression.operator_metrics import (
    adapt_candidates_for_model,
    build_info_dict_from_cache,
    compute_model_costs,
    expand_info_for_candidates,
    load_model_from_path,
    maybe_align_action_width,
)
from oawc.models import load_cost_model


@dataclass
class LoadedOperatorCache:
    path: Path
    payload: dict[str, Any]
    candidate_actions: torch.Tensor
    teacher_costs: torch.Tensor
    teacher_best_index: torch.Tensor
    teacher_best_first_action: torch.Tensor


def load_operator_cache(
    cache_path: str | Path,
    device: str = "cpu",
) -> LoadedOperatorCache:
    del device  # cache tensors remain on CPU
    path = Path(cache_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return LoadedOperatorCache(
        path=path,
        payload=payload,
        candidate_actions=payload["candidate_actions"].float(),
        teacher_costs=payload["teacher_costs"].float(),
        teacher_best_index=payload["teacher_best_index"].long(),
        teacher_best_first_action=payload["teacher_best_first_action"].float(),
    )


def _stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(x.min()),
        "mean": float(x.mean()),
        "max": float(x.max()),
        "std": float(x.std()),
    }


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    idx = torch.argsort(x)
    ranks = torch.empty_like(idx, dtype=torch.float32)
    ranks[idx] = torch.arange(
        x.numel(),
        dtype=torch.float32,
        device=x.device,
    )
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx.pow(2).sum()) * (ry.pow(2).sum()))
    if float(denom) == 0.0:
        return 0.0
    return float((rx * ry).sum().item() / denom.item())


def _zscore_per_state(costs: torch.Tensor) -> torch.Tensor:
    mu = costs.mean(dim=1, keepdim=True)
    std = costs.std(dim=1, keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (costs - mu) / std


def compute_operator_metrics(
    *,
    teacher_costs: torch.Tensor,
    student_costs: torch.Tensor,
    candidate_actions: torch.Tensor,
    teacher_best_index: torch.Tensor | None = None,
    teacher_best_first_action: torch.Tensor | None = None,
    topk: list[int] | None = None,
) -> dict[str, Any]:
    teacher_costs = teacher_costs.detach().cpu().float()
    student_costs = student_costs.detach().cpu().float()
    candidate_actions = candidate_actions.detach().cpu().float()
    if teacher_best_index is None:
        teacher_best_index = torch.argmin(teacher_costs, dim=1).long()
    else:
        teacher_best_index = teacher_best_index.detach().cpu().long()
    if teacher_best_first_action is None:
        teacher_best_first_action = candidate_actions[
            torch.arange(candidate_actions.shape[0]),
            teacher_best_index,
            0,
            :,
        ]
    else:
        teacher_best_first_action = (
            teacher_best_first_action.detach().cpu().float()
        )

    finite_student = bool(torch.isfinite(student_costs).all().item())
    raw_cost_mse = float(
        torch.mean((student_costs - teacher_costs) ** 2).item()
    )
    t_norm = _zscore_per_state(teacher_costs)
    s_norm = _zscore_per_state(student_costs)
    norm_mse_per_state = ((t_norm - s_norm) ** 2).mean(dim=1)
    spearman_per_state = torch.tensor(
        [
            _spearman(teacher_costs[i], student_costs[i])
            for i in range(teacher_costs.shape[0])
        ],
        dtype=torch.float32,
    )

    teacher_sorted = torch.argsort(teacher_costs, dim=1)
    student_sorted = torch.argsort(student_costs, dim=1)
    topk_vals = (
        sorted(int(k) for k in topk) if topk is not None else [1, 5, 10, 20]
    )
    topk_overlap: dict[str, dict[str, float]] = {}
    for k in topk_vals:
        kk = int(min(k, teacher_costs.shape[1]))
        overlaps = []
        for i in range(teacher_costs.shape[0]):
            t_set = set(teacher_sorted[i, :kk].tolist())
            s_set = set(student_sorted[i, :kk].tolist())
            overlaps.append(len(t_set.intersection(s_set)) / kk)
        topk_overlap[str(k)] = _stats(
            torch.tensor(overlaps, dtype=torch.float32)
        )

    student_best_index = student_sorted[:, 0]
    teacher_best_cost = teacher_costs[
        torch.arange(teacher_costs.shape[0]),
        teacher_best_index,
    ]
    student_pick_teacher_cost = teacher_costs[
        torch.arange(teacher_costs.shape[0]),
        student_best_index,
    ]
    teacher_regret = student_pick_teacher_cost - teacher_best_cost
    student_best_first_action = candidate_actions[
        torch.arange(candidate_actions.shape[0]),
        student_best_index,
        0,
        :,
    ]
    first_action_error = torch.linalg.norm(
        student_best_first_action - teacher_best_first_action,
        dim=1,
    )
    match_rate = float(
        (student_best_index == teacher_best_index).float().mean().item()
    )

    return {
        "finite_student_costs": finite_student,
        "raw_cost_mse": raw_cost_mse,
        "normalized_cost_mse_per_state": _stats(norm_mse_per_state),
        "spearman_per_state": _stats(spearman_per_state),
        "topk_overlap": topk_overlap,
        "teacher_regret": _stats(teacher_regret),
        "selected_first_action_error": _stats(first_action_error),
        "teacher_best_index_match_rate": match_rate,
        "student_best_index": student_best_index.tolist(),
        "teacher_best_index": teacher_best_index.tolist(),
        "teacher_cost_stats": _stats(teacher_costs),
        "student_cost_stats": _stats(student_costs),
    }


def _slice_info_dict(
    info_dict: dict[str, torch.Tensor],
    state_slice: slice,
    device: str,
) -> dict[str, torch.Tensor]:
    return {k: v[state_slice].to(device) for k, v in info_dict.items()}


def _compute_costs_chunked(
    *,
    model: torch.nn.Module,
    info_cpu: dict[str, torch.Tensor],
    candidate_actions_cpu: torch.Tensor,
    device: str,
    batch_states: int,
    batch_candidates: int,
) -> torch.Tensor:
    num_states = int(candidate_actions_cpu.shape[0])
    num_candidates = int(candidate_actions_cpu.shape[1])
    costs_cpu = torch.empty((num_states, num_candidates), dtype=torch.float32)
    for s0 in range(0, num_states, batch_states):
        s1 = min(num_states, s0 + batch_states)
        info_chunk = _slice_info_dict(info_cpu, slice(s0, s1), device)
        for c0 in range(0, num_candidates, batch_candidates):
            c1 = min(num_candidates, c0 + batch_candidates)
            cand_chunk = candidate_actions_cpu[s0:s1, c0:c1].to(device)
            cand_eval = adapt_candidates_for_model(cand_chunk, model)
            expanded = expand_info_for_candidates(
                info_chunk,
                num_candidates=int(c1 - c0),
            )
            with torch.no_grad():
                out = compute_model_costs(model, expanded, cand_eval)
            costs_cpu[s0:s1, c0:c1] = out.detach().to("cpu")
            del cand_chunk, cand_eval, expanded, out
        del info_chunk
    return costs_cpu


def _load_teacher_model_from_cache(
    *,
    cache_payload: dict[str, Any],
    env_name: str,
    device: str,
) -> torch.nn.Module:
    family = str(cache_payload["model_family"])
    checkpoint = str(
        cache_payload.get("resolved_checkpoint", cache_payload["checkpoint"])
    )
    loaded = load_cost_model(
        family=family,
        checkpoint=checkpoint,
        env_name=env_name,
        device=device,
    )
    model = loaded.model.to(device).eval()
    model.requires_grad_(False)
    return model


def evaluate_model_on_operator_cache(
    *,
    cache_path: str | Path | None = None,
    cache: LoadedOperatorCache | None = None,
    model_path: str | Path | None = None,
    model: torch.nn.Module | None = None,
    env_name: str | None = None,
    device: str = "cpu",
    batch_states: int = 8,
    batch_candidates: int = 128,
    state_indices: list[int] | None = None,
    candidate_actions_override: torch.Tensor | None = None,
    teacher_costs_override: torch.Tensor | None = None,
    teacher_best_index_override: torch.Tensor | None = None,
    teacher_best_first_action_override: torch.Tensor | None = None,
    teacher_model: torch.nn.Module | None = None,
    use_chunked_student: bool = False,
    topk: list[int] | None = None,
) -> dict[str, Any]:
    if cache is None:
        if cache_path is None:
            raise ValueError("Provide cache or cache_path.")
        cache = load_operator_cache(cache_path)
    if model is None:
        if model_path is None:
            raise ValueError("Provide model or model_path.")
        model = load_model_from_path(model_path, device=device).eval()
        model.requires_grad_(False)

    payload = cache.payload
    env = env_name or str(payload["env"])
    if state_indices is None:
        state_idx = torch.arange(
            cache.candidate_actions.shape[0],
            dtype=torch.long,
        )
    else:
        state_idx = torch.as_tensor(state_indices, dtype=torch.long)

    candidate_actions = (
        cache.candidate_actions[state_idx]
        if candidate_actions_override is None
        else candidate_actions_override.float()
    )
    if int(candidate_actions.shape[0]) != int(state_idx.shape[0]):
        raise ValueError(
            (
                "candidate_actions first dimension must match selected "
                "state count."
            )
        )

    candidate_mode = (
        "exact" if candidate_actions_override is None else "generated"
    )
    teacher_source = "cache"
    teacher_costs = (
        cache.teacher_costs[state_idx]
        if teacher_costs_override is None
        else teacher_costs_override.float()
    )
    teacher_best_index = (
        cache.teacher_best_index[state_idx]
        if teacher_best_index_override is None
        else teacher_best_index_override.long()
    )
    teacher_best_first_action = (
        cache.teacher_best_first_action[state_idx]
        if teacher_best_first_action_override is None
        else teacher_best_first_action_override.float()
    )

    if (
        candidate_actions_override is not None
        and teacher_costs_override is None
    ):
        teacher_source = "recomputed"
        tm = teacher_model or _load_teacher_model_from_cache(
            cache_payload=payload,
            env_name=env,
            device=device,
        )
        info_cpu = build_info_dict_from_cache(
            env_name=env,
            episodes_idx=[
                payload["episodes_idx"][int(i)] for i in state_idx.tolist()
            ],
            start_steps=[
                payload["start_steps"][int(i)] for i in state_idx.tolist()
            ],
            goal_offset_steps=int(payload["goal_offset_steps"]),
            device="cpu",
        )
        info_cpu = maybe_align_action_width(info_cpu, tm)
        teacher_costs = _compute_costs_chunked(
            model=tm,
            info_cpu=info_cpu,
            candidate_actions_cpu=candidate_actions,
            device=device,
            batch_states=batch_states,
            batch_candidates=batch_candidates,
        )
        teacher_best_index = torch.argmin(teacher_costs, dim=1).long()
        teacher_best_first_action = candidate_actions[
            torch.arange(candidate_actions.shape[0]),
            teacher_best_index,
            0,
            :,
        ]

    info = build_info_dict_from_cache(
        env_name=env,
        episodes_idx=[
            payload["episodes_idx"][int(i)] for i in state_idx.tolist()
        ],
        start_steps=[
            payload["start_steps"][int(i)] for i in state_idx.tolist()
        ],
        goal_offset_steps=int(payload["goal_offset_steps"]),
        device=device,
    )
    info = maybe_align_action_width(info, model)
    candidate_eval = adapt_candidates_for_model(
        candidate_actions.to(device),
        model,
    )
    if use_chunked_student:
        info_cpu = {k: v.detach().cpu() for k, v in info.items()}
        student_costs = _compute_costs_chunked(
            model=model,
            info_cpu=info_cpu,
            candidate_actions_cpu=candidate_actions,
            device=device,
            batch_states=batch_states,
            batch_candidates=batch_candidates,
        )
    else:
        expanded = expand_info_for_candidates(
            info,
            num_candidates=int(candidate_actions.shape[1]),
        )
        with torch.no_grad():
            student = compute_model_costs(model, expanded, candidate_eval)
        student_costs = student.detach().to("cpu").float()

    metrics = compute_operator_metrics(
        teacher_costs=teacher_costs,
        student_costs=student_costs,
        candidate_actions=candidate_actions,
        teacher_best_index=teacher_best_index,
        teacher_best_first_action=teacher_best_first_action,
        topk=topk,
    )
    return {
        "teacher_costs": teacher_costs,
        "student_costs": student_costs,
        "candidate_actions": candidate_actions,
        "teacher_best_index": teacher_best_index,
        "teacher_best_first_action": teacher_best_first_action,
        "metrics": metrics,
        "metadata": {
            "cache_path": str(cache.path),
            "model_path": str(model_path) if model_path is not None else None,
            "device": device,
            "state_count": int(state_idx.shape[0]),
            "candidate_count": int(candidate_actions.shape[1]),
            "candidate_mode": candidate_mode,
            "teacher_source": teacher_source,
            "use_chunked_student": bool(use_chunked_student),
            "candidate_actions_raw_shape": list(candidate_actions.shape),
            "candidate_actions_eval_shape": list(candidate_eval.shape),
            "teacher_costs_shape": list(teacher_costs.shape),
            "student_costs_shape": list(student_costs.shape),
        },
    }
