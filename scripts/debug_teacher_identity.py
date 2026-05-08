from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from oawc.compression.operator_eval import load_operator_cache
from oawc.compression.operator_metrics import (
    _compute_cost_with_jepa_fallback,
    adapt_candidates_for_model,
    build_info_dict_from_cache,
    maybe_align_action_width,
    resolve_device,
)
from oawc.compression.reports import save_json
from oawc.models import load_cost_model


def _stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "max": float(x.max().item()),
        "std": float(x.std().item()),
    }


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    idx = torch.argsort(x)
    ranks = torch.empty_like(idx, dtype=torch.float32)
    ranks[idx] = torch.arange(x.numel(), dtype=torch.float32, device=x.device)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx.pow(2).sum()) * (ry.pow(2).sum()))
    if float(denom.item()) == 0.0:
        return 0.0
    return float((rx * ry).sum().item() / denom.item())


def _pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    mse = float(torch.mean((a - b) ** 2).item())
    spearman_mean = float(
        torch.tensor(
            [_spearman(a[i], b[i]) for i in range(a.shape[0])],
            dtype=torch.float32,
        ).mean().item()
    )
    top1_match = float(
        (torch.argmin(a, dim=1) == torch.argmin(b, dim=1)).float().mean().item()
    )
    return {
        "mse": mse,
        "spearman_mean": spearman_mean,
        "top1_match_rate": top1_match,
    }


def _first_divergence(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float = 1e-6,
) -> dict[str, Any] | None:
    diff = (a - b).abs()
    bad = diff > atol
    if not bool(bad.any().item()):
        return None
    ij = torch.nonzero(bad, as_tuple=False)[0]
    i = int(ij[0].item())
    j = int(ij[1].item())
    return {
        "state_index": i,
        "candidate_index": j,
        "a_value": float(a[i, j].item()),
        "b_value": float(b[i, j].item()),
        "abs_diff": float(diff[i, j].item()),
    }


def _slice_info_dict(
    info: dict[str, torch.Tensor],
    state_slice: slice,
    device: str,
) -> dict[str, torch.Tensor]:
    return {k: v[state_slice].to(device) for k, v in info.items()}


def _expand_info_for_candidates(
    info: dict[str, torch.Tensor],
    num_candidates: int,
) -> dict[str, torch.Tensor]:
    return {
        k: v.unsqueeze(1).repeat(1, num_candidates, *([1] * (v.ndim - 1)))
        for k, v in info.items()
    }


def _compute_costs_with_trace(
    *,
    model: torch.nn.Module,
    info_cpu: dict[str, torch.Tensor],
    candidate_actions_cpu: torch.Tensor,
    candidate_actions_eval_cpu: torch.Tensor,
    device: str,
    batch_states: int,
    batch_candidates: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    num_states = int(candidate_actions_eval_cpu.shape[0])
    num_candidates = int(candidate_actions_eval_cpu.shape[1])
    out = torch.empty((num_states, num_candidates), dtype=torch.float32)
    trace = {
        "dispatch_order": ["get_cost", "fallback_jepa"],
        "model_has_get_cost": bool(hasattr(model, "get_cost")),
        "model_has_cost": bool(hasattr(model, "cost")),
        "model_has_forward": bool(hasattr(model, "forward")),
        "branch_counts": {
            "get_cost": 0,
            "fallback_jepa": 0,
        },
    }
    for s0 in range(0, num_states, batch_states):
        s1 = min(num_states, s0 + batch_states)
        info_chunk = _slice_info_dict(info_cpu, slice(s0, s1), device)
        for c0 in range(0, num_candidates, batch_candidates):
            c1 = min(num_candidates, c0 + batch_candidates)
            cand_eval = candidate_actions_eval_cpu[s0:s1, c0:c1].to(device)
            expanded = _expand_info_for_candidates(info_chunk, int(c1 - c0))
            with torch.no_grad():
                try:
                    costs = model.get_cost(expanded, cand_eval)
                    trace["branch_counts"]["get_cost"] += 1
                except RuntimeError as exc:
                    msg = str(exc)
                    if (
                        "expanded size of the tensor" in msg
                        and hasattr(model, "rollout")
                        and hasattr(model, "encode")
                    ):
                        costs = _compute_cost_with_jepa_fallback(
                            model,
                            expanded,
                            cand_eval,
                        )
                        trace["branch_counts"]["fallback_jepa"] += 1
                    else:
                        raise
            out[s0:s1, c0:c1] = costs.detach().cpu()
    trace["candidate_actions_raw_shape"] = list(candidate_actions_cpu.shape)
    trace["candidate_actions_eval_shape"] = list(candidate_actions_eval_cpu.shape)
    return out, trace


def _evaluate_path(
    *,
    model: torch.nn.Module,
    env_name: str,
    episodes_idx: list[int],
    start_steps: list[int],
    goal_offset_steps: int,
    candidate_actions: torch.Tensor,
    device: str,
    batch_states: int,
    batch_candidates: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    info_raw = build_info_dict_from_cache(
        env_name=env_name,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset_steps=goal_offset_steps,
        device="cpu",
    )
    info_aligned = maybe_align_action_width(dict(info_raw), model)
    candidate_eval = adapt_candidates_for_model(candidate_actions.to(device), model)
    costs, trace = _compute_costs_with_trace(
        model=model,
        info_cpu=info_aligned,
        candidate_actions_cpu=candidate_actions,
        candidate_actions_eval_cpu=candidate_eval.detach().cpu(),
        device=device,
        batch_states=batch_states,
        batch_candidates=batch_candidates,
    )
    info_raw_shapes = {k: list(v.shape) for k, v in info_raw.items()}
    info_aligned_shapes = {k: list(v.shape) for k, v in info_aligned.items()}
    meta = {
        "info_raw_keys": sorted(info_raw.keys()),
        "info_raw_shapes": info_raw_shapes,
        "info_aligned_shapes": info_aligned_shapes,
        "candidate_actions_shape_before_adapt": list(candidate_actions.shape),
        "candidate_actions_shape_after_adapt": list(candidate_eval.shape),
        "compute_dispatch_trace": trace,
    }
    return costs, meta


def _load_model_from_local(path: str, device: str) -> torch.nn.Module:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.nn.Module):
        model = loaded
    else:
        model = getattr(loaded, "model", loaded)
    model = model.to(device).eval()
    model.requires_grad_(False)
    if not hasattr(model, "get_cost"):
        raise TypeError("Local model does not expose get_cost().")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--hf-checkpoint", default="quentinll/lewm-tworooms")
    parser.add_argument("--hf-family", default="lewm_hf")
    parser.add_argument("--local-model-path", required=True)
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-candidates", type=int, default=128)
    parser.add_argument("--tag", default="teacher_identity")
    args = parser.parse_args()

    device = resolve_device(args.device)
    cache = load_operator_cache(args.cache)
    env = str(cache.payload["env"])
    candidate_actions = cache.candidate_actions.float()
    cached_teacher_costs = cache.teacher_costs.float()

    hf_loaded = load_cost_model(
        family=args.hf_family,
        checkpoint=args.hf_checkpoint,
        env_name=env,
        device=device,
    )
    hf_model = hf_loaded.model.to(device).eval()
    hf_model.requires_grad_(False)

    local_model = _load_model_from_local(args.local_model_path, device)

    hf_costs, hf_meta = _evaluate_path(
        model=hf_model,
        env_name=env,
        episodes_idx=list(cache.payload["episodes_idx"]),
        start_steps=list(cache.payload["start_steps"]),
        goal_offset_steps=int(cache.payload["goal_offset_steps"]),
        candidate_actions=candidate_actions,
        device=device,
        batch_states=max(1, int(args.batch_states)),
        batch_candidates=max(1, int(args.batch_candidates)),
    )
    local_costs, local_meta = _evaluate_path(
        model=local_model,
        env_name=env,
        episodes_idx=list(cache.payload["episodes_idx"]),
        start_steps=list(cache.payload["start_steps"]),
        goal_offset_steps=int(cache.payload["goal_offset_steps"]),
        candidate_actions=candidate_actions,
        device=device,
        batch_states=max(1, int(args.batch_states)),
        batch_candidates=max(1, int(args.batch_candidates)),
    )

    cache_vs_hf = _pair_metrics(cached_teacher_costs, hf_costs)
    cache_vs_local = _pair_metrics(cached_teacher_costs, local_costs)
    hf_vs_local = _pair_metrics(hf_costs, local_costs)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cache_path": str(Path(args.cache)),
        "hf_checkpoint": args.hf_checkpoint,
        "local_model_path": str(Path(args.local_model_path)),
        "device": device,
        "candidate_actions_shape": list(candidate_actions.shape),
        "candidate_actions_first_values": [
            float(v)
            for v in candidate_actions.reshape(-1)[:16].tolist()
        ],
        "teacher_costs_cache_shape": list(cached_teacher_costs.shape),
        "teacher_costs_hf_shape": list(hf_costs.shape),
        "teacher_costs_local_shape": list(local_costs.shape),
        "teacher_costs_cache_stats": _stats(cached_teacher_costs),
        "teacher_costs_hf_stats": _stats(hf_costs),
        "teacher_costs_local_stats": _stats(local_costs),
        "cache_vs_hf": {
            **cache_vs_hf,
            "first_divergence": _first_divergence(cached_teacher_costs, hf_costs),
        },
        "cache_vs_local": {
            **cache_vs_local,
            "first_divergence": _first_divergence(cached_teacher_costs, local_costs),
        },
        "hf_vs_local": {
            **hf_vs_local,
            "first_divergence": _first_divergence(hf_costs, local_costs),
        },
        "hf_path_debug": hf_meta,
        "local_path_debug": local_meta,
    }

    out_path = Path("outputs/tables") / f"debug_teacher_identity_{args.tag}.json"
    save_json(out_path, report)
    print("Teacher identity debug written")
    print(f"  output: {out_path}")
    print(
        "  cache vs hf:    mse={:.6f} spearman={:.6f} top1={:.6f}".format(
            report["cache_vs_hf"]["mse"],
            report["cache_vs_hf"]["spearman_mean"],
            report["cache_vs_hf"]["top1_match_rate"],
        )
    )
    print(
        "  cache vs local: mse={:.6f} spearman={:.6f} top1={:.6f}".format(
            report["cache_vs_local"]["mse"],
            report["cache_vs_local"]["spearman_mean"],
            report["cache_vs_local"]["top1_match_rate"],
        )
    )
    print(
        "  hf vs local:    mse={:.6f} spearman={:.6f} top1={:.6f}".format(
            report["hf_vs_local"]["mse"],
            report["hf_vs_local"]["spearman_mean"],
            report["hf_vs_local"]["top1_match_rate"],
        )
    )


if __name__ == "__main__":
    main()
