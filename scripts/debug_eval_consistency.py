from __future__ import annotations

import argparse
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
    resolve_device,
)
from oawc.compression.reports import save_json


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    idx_x = torch.argsort(x)
    idx_y = torch.argsort(y)
    rx = torch.empty_like(idx_x, dtype=torch.float32)
    ry = torch.empty_like(idx_y, dtype=torch.float32)
    rx[idx_x] = torch.arange(
        x.numel(),
        dtype=torch.float32,
        device=x.device,
    )
    ry[idx_y] = torch.arange(
        y.numel(),
        dtype=torch.float32,
        device=y.device,
    )
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx.pow(2).sum()) * (ry.pow(2).sum()))
    if float(denom.item()) == 0.0:
        return 0.0
    return float((rx * ry).sum().item() / denom.item())


def _topk_overlap_mean(
    teacher_sorted: torch.Tensor,
    student_sorted: torch.Tensor,
    k: int,
) -> float:
    overlaps = []
    for i in range(teacher_sorted.shape[0]):
        t_set = set(teacher_sorted[i, :k].tolist())
        s_set = set(student_sorted[i, :k].tolist())
        overlaps.append(len(t_set.intersection(s_set)) / float(k))
    return float(torch.tensor(overlaps, dtype=torch.float32).mean().item())


def _slice_info_dict(
    info_dict: dict[str, torch.Tensor],
    state_slice: slice,
    device: str,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in info_dict.items():
        out[key] = value[state_slice].to(device)
    return out


def _compute_student_costs_chunked(
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
        state_slice = slice(s0, s1)
        info_chunk = _slice_info_dict(info_cpu, state_slice, device)
        for c0 in range(0, num_candidates, batch_candidates):
            c1 = min(num_candidates, c0 + batch_candidates)
            cand_chunk = candidate_actions_cpu[state_slice, c0:c1].to(device)
            cand_eval = adapt_candidates_for_model(cand_chunk, model)
            expanded = expand_info_for_candidates(
                info_chunk,
                num_candidates=int(c1 - c0),
            )
            with torch.no_grad():
                s_cost = compute_model_costs(model, expanded, cand_eval)
            costs_cpu[state_slice, c0:c1] = s_cost.detach().to("cpu")
            del cand_chunk
            del cand_eval
            del expanded
            del s_cost
        del info_chunk
    return costs_cpu


def _aggregate_metrics(
    *,
    teacher_costs_cpu: torch.Tensor,
    student_costs_cpu: torch.Tensor,
    candidate_actions_cpu: torch.Tensor,
    teacher_best_index_cpu: torch.Tensor,
    teacher_best_first_action_cpu: torch.Tensor,
) -> dict[str, Any]:
    teacher_sorted = torch.argsort(teacher_costs_cpu, dim=1)
    student_sorted = torch.argsort(student_costs_cpu, dim=1)
    student_best = student_sorted[:, 0]
    teacher_best = teacher_sorted[:, 0]
    regret = (
        teacher_costs_cpu[
            torch.arange(teacher_costs_cpu.shape[0]),
            student_best,
        ]
        - teacher_costs_cpu[
            torch.arange(teacher_costs_cpu.shape[0]),
            teacher_best,
        ]
    )
    student_best_first_action = candidate_actions_cpu[
        torch.arange(candidate_actions_cpu.shape[0]),
        student_best,
        0,
        :,
    ]
    first_action_error = torch.linalg.norm(
        student_best_first_action - teacher_best_first_action_cpu,
        dim=1,
    )
    spearman_per_state = torch.tensor(
        [
            _spearman(teacher_costs_cpu[i], student_costs_cpu[i])
            for i in range(teacher_costs_cpu.shape[0])
        ],
        dtype=torch.float32,
    )
    return {
        "spearman": float(spearman_per_state.mean().item()),
        "top1_overlap": _topk_overlap_mean(teacher_sorted, student_sorted, 1),
        "top5_overlap": _topk_overlap_mean(teacher_sorted, student_sorted, 5),
        "top10_overlap": _topk_overlap_mean(
            teacher_sorted,
            student_sorted,
            10,
        ),
        "regret": float(regret.mean().item()),
        "first_action_error": float(first_action_error.mean().item()),
        "teacher_best_index_match_rate": float(
            (student_best == teacher_best_index_cpu).float().mean().item()
        ),
        "spearman_per_state_first3": [
            float(v)
            for v in spearman_per_state[:3].tolist()
        ],
    }


def _old_eval_path(
    *,
    model: torch.nn.Module,
    cache: dict[str, Any],
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    info = build_info_dict_from_cache(
        env_name=cache["env"],
        episodes_idx=list(cache["episodes_idx"]),
        start_steps=list(cache["start_steps"]),
        goal_offset_steps=int(cache["goal_offset_steps"]),
        device=device,
    )
    info = maybe_align_action_width(info, model)
    candidate_actions_cpu = cache["candidate_actions"].float()
    candidate_eval = adapt_candidates_for_model(
        candidate_actions_cpu.to(device),
        model,
    )
    expanded = expand_info_for_candidates(
        info,
        num_candidates=int(candidate_actions_cpu.shape[1]),
    )
    with torch.no_grad():
        student_costs = compute_model_costs(model, expanded, candidate_eval)
    student_costs_cpu = student_costs.detach().to("cpu").float()
    metadata = {
        "candidate_actions_raw_shape": list(candidate_actions_cpu.shape),
        "candidate_actions_eval_shape": list(candidate_eval.shape),
        "student_costs_shape": list(student_costs_cpu.shape),
    }
    return student_costs_cpu, metadata


def _crossed_exact_eval_path(
    *,
    model: torch.nn.Module,
    cache: dict[str, Any],
    device: str,
    batch_states: int,
    batch_candidates: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    info_cpu = build_info_dict_from_cache(
        env_name=cache["env"],
        episodes_idx=list(cache["episodes_idx"]),
        start_steps=list(cache["start_steps"]),
        goal_offset_steps=int(cache["goal_offset_steps"]),
        device="cpu",
    )
    info_cpu = maybe_align_action_width(info_cpu, model)
    candidate_actions_cpu = cache["candidate_actions"].float()
    student_costs_cpu = _compute_student_costs_chunked(
        model=model,
        info_cpu=info_cpu,
        candidate_actions_cpu=candidate_actions_cpu,
        device=device,
        batch_states=batch_states,
        batch_candidates=batch_candidates,
    )
    metadata = {
        "candidate_actions_raw_shape": list(candidate_actions_cpu.shape),
        "candidate_actions_eval_shape": list(
            adapt_candidates_for_model(
                candidate_actions_cpu[:1, :1].to(device),
                model,
            ).shape
        ),
        "student_costs_shape": list(student_costs_cpu.shape),
    }
    return student_costs_cpu, metadata


def _first_divergence(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float,
) -> dict[str, Any] | None:
    diff = (a - b).abs()
    bad = diff > atol
    if not bool(bad.any().item()):
        return None
    idx = torch.nonzero(bad, as_tuple=False)[0]
    i = int(idx[0].item())
    j = int(idx[1].item())
    return {
        "state_index": i,
        "candidate_index": j,
        "old_value": float(a[i, j].item()),
        "crossed_value": float(b[i, j].item()),
        "abs_diff": float(diff[i, j].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-candidates", type=int, default=128)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    model_path = Path(args.model_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache: {cache_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    device = resolve_device(args.device)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    model = load_model_from_path(model_path, device=device).eval()
    model.requires_grad_(False)

    teacher_costs = cache["teacher_costs"].float()
    candidate_actions = cache["candidate_actions"].float()
    teacher_best_index = cache["teacher_best_index"].long()
    teacher_best_first_action = cache["teacher_best_first_action"].float()

    old_costs, old_meta = _old_eval_path(
        model=model,
        cache=cache,
        device=device,
    )
    crossed_costs, crossed_meta = _crossed_exact_eval_path(
        model=model,
        cache=cache,
        device=device,
        batch_states=int(args.batch_states),
        batch_candidates=int(args.batch_candidates),
    )

    old_metrics = _aggregate_metrics(
        teacher_costs_cpu=teacher_costs,
        student_costs_cpu=old_costs,
        candidate_actions_cpu=candidate_actions,
        teacher_best_index_cpu=teacher_best_index,
        teacher_best_first_action_cpu=teacher_best_first_action,
    )
    crossed_metrics = _aggregate_metrics(
        teacher_costs_cpu=teacher_costs,
        student_costs_cpu=crossed_costs,
        candidate_actions_cpu=candidate_actions,
        teacher_best_index_cpu=teacher_best_index,
        teacher_best_first_action_cpu=teacher_best_first_action,
    )

    exact_equal = bool(torch.equal(old_costs, crossed_costs))
    allclose = bool(
        torch.allclose(
            old_costs,
            crossed_costs,
            atol=float(args.atol),
            rtol=0.0,
        )
    )
    first_div = _first_divergence(old_costs, crossed_costs, float(args.atol))

    payload = {
        "cache_path": str(cache_path),
        "model_path": str(model_path),
        "device": device,
        "cache_shapes": {
            "candidate_actions": list(candidate_actions.shape),
            "teacher_costs": list(teacher_costs.shape),
            "teacher_best_index": list(teacher_best_index.shape),
            "teacher_best_first_action": list(teacher_best_first_action.shape),
        },
        "old_eval_path": {
            "metadata": old_meta,
            "aggregate": old_metrics,
            "teacher_costs_first3x8": [
                [float(v) for v in row]
                for row in teacher_costs[:3, :8].tolist()
            ],
            "student_costs_first3x8": [
                [float(v) for v in row]
                for row in old_costs[:3, :8].tolist()
            ],
        },
        "crossed_eval_exact_path": {
            "metadata": crossed_meta,
            "aggregate": crossed_metrics,
            "teacher_costs_first3x8": [
                [float(v) for v in row]
                for row in teacher_costs[:3, :8].tolist()
            ],
            "student_costs_first3x8": [
                [float(v) for v in row]
                for row in crossed_costs[:3, :8].tolist()
            ],
        },
        "comparison": {
            "costs_exact_equal": exact_equal,
            "costs_allclose": allclose,
            "max_abs_diff": float(
                (old_costs - crossed_costs).abs().max().item()
            ),
            "first_divergence": first_div,
        },
    }

    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{model_path.parent.name}_{cache_path.parent.name}"
    out_path = out_dir / f"debug_eval_consistency_{tag}.json"
    save_json(out_path, payload)

    print("Debug eval consistency report written")
    print(f"  model_path: {model_path}")
    print(f"  cache_path: {cache_path}")
    print(f"  output:     {out_path}")
    print("  old aggregate:")
    print(
        "    spearman={:.6f} top5={:.6f} regret={:.6f}".format(
            old_metrics["spearman"],
            old_metrics["top5_overlap"],
            old_metrics["regret"],
        )
    )
    print("  crossed aggregate:")
    print(
        "    spearman={:.6f} top5={:.6f} regret={:.6f}".format(
            crossed_metrics["spearman"],
            crossed_metrics["top5_overlap"],
            crossed_metrics["regret"],
        )
    )
    print(
        "  comparison: exact_equal={} allclose={} max_abs_diff={:.8f}".format(
            exact_equal,
            allclose,
            payload["comparison"]["max_abs_diff"],
        )
    )
    if first_div is not None:
        print(
            (
                "  first divergence: state={} cand={} old={:.6f} "
                "crossed={:.6f} diff={:.6f}"
            ).format(
                first_div["state_index"],
                first_div["candidate_index"],
                first_div["old_value"],
                first_div["crossed_value"],
                first_div["abs_diff"],
            )
        )


if __name__ == "__main__":
    main()
