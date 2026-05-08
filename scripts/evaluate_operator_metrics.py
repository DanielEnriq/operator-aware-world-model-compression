from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from oawc.compression.operator_metrics import (
    adapt_candidates_for_model,
    build_info_dict_from_cache,
    compute_model_costs,
    expand_info_for_candidates,
    load_model_from_path,
    maybe_align_action_width,
    resolve_device,
)
from oawc.compression.reports import save_json


def _stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(x.min()),
        "mean": float(x.mean()),
        "max": float(x.max()),
        "std": float(x.std()),
    }


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    # Ordinal ranks, deterministic for our almost-always continuous costs.
    idx = torch.argsort(x)
    ranks = torch.empty_like(idx, dtype=torch.float32)
    ranks[idx] = torch.arange(x.numel(), dtype=torch.float32)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    env_name = cache["env"]

    device = resolve_device(args.device)
    model = load_model_from_path(args.model_path, device=device)

    teacher_costs = cache["teacher_costs"].float()
    candidate_actions = cache["candidate_actions"].float()
    teacher_best_index = cache["teacher_best_index"].long()
    teacher_best_first_action = cache["teacher_best_first_action"].float()
    topk = sorted(int(k) for k in cache.get("topk_indices", {}).keys())

    info_dict = build_info_dict_from_cache(
        env_name=env_name,
        episodes_idx=list(cache["episodes_idx"]),
        start_steps=list(cache["start_steps"]),
        goal_offset_steps=int(cache["goal_offset_steps"]),
        device=device,
    )
    info_dict = maybe_align_action_width(info_dict, model)

    candidate_actions_eval = adapt_candidates_for_model(
        candidate_actions.to(device),
        model,
    )
    expanded_info = expand_info_for_candidates(
        info_dict,
        num_candidates=int(candidate_actions.shape[1]),
    )

    with torch.no_grad():
        student_costs = compute_model_costs(
            model,
            expanded_info,
            candidate_actions_eval,
        )
    student_costs = student_costs.detach().cpu().float()

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

    topk_overlap: dict[str, dict[str, float]] = {}
    for k in topk:
        overlaps = []
        for i in range(teacher_costs.shape[0]):
            t_set = set(teacher_sorted[i, :k].tolist())
            s_set = set(student_sorted[i, :k].tolist())
            overlaps.append(len(t_set.intersection(s_set)) / k)
        overlap_t = torch.tensor(overlaps, dtype=torch.float32)
        topk_overlap[str(k)] = _stats(overlap_t)

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

    best_index_match = student_best_index == teacher_best_index
    match_rate = float(best_index_match.float().mean().item())

    output_dir = Path("outputs/operator_metrics") / env_name / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics.json"

    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "env": env_name,
        "tag": args.tag,
        "cache_path": str(cache_path),
        "model_path": str(Path(args.model_path)),
        "device": device,
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
    save_json(out_path, metrics)

    print("Operator metrics evaluation complete")
    print(f"  tag:                         {args.tag}")
    print(f"  finite student costs:        {finite_student}")
    print(f"  raw cost mse:                {raw_cost_mse:.6f}")
    print(f"  teacher best index match:    {match_rate:.6f}")
    print(
        "  spearman mean:               "
        f"{metrics['spearman_per_state']['mean']:.6f}"
    )
    print(f"  saved:                       {out_path}")


if __name__ == "__main__":
    main()
