from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from oawc.compression.operator_metrics import build_info_dict_from_cache
from oawc.compression.reports import save_json


def _stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1)
    if x.numel() == 0:
        return {
            "count": 0.0,
            "min": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
    q = torch.quantile(
        x,
        torch.tensor(
            [0.10, 0.25, 0.50, 0.75, 0.90],
            dtype=torch.float32,
            device=x.device,
        ),
    )
    return {
        "count": float(x.numel()),
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "max": float(x.max().item()),
        "std": float(x.std().item()),
        "p10": float(q[0].item()),
        "p25": float(q[1].item()),
        "p50": float(q[2].item()),
        "p75": float(q[3].item()),
        "p90": float(q[4].item()),
    }


def _shape(x: Any) -> list[int] | None:
    if torch.is_tensor(x):
        return list(x.shape)
    return None


def _best_index_entropy(
    best_idx: torch.Tensor,
    num_candidates: int,
) -> dict[str, Any]:
    if best_idx.numel() == 0:
        return {
            "entropy_bits": float("nan"),
            "unique_count": 0,
            "distribution": [],
        }
    counts = torch.bincount(best_idx.long(), minlength=num_candidates).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nz = probs > 0
    entropy = float((-(probs[nz] * torch.log2(probs[nz]))).sum().item())
    dist = [
        {"index": int(i), "count": int(c.item()), "prob": float(p.item())}
        for i, (c, p) in enumerate(zip(counts, probs))
        if c.item() > 0
    ]
    dist.sort(key=lambda d: d["count"], reverse=True)
    return {
        "entropy_bits": entropy,
        "unique_count": int((counts > 0).sum().item()),
        "distribution": dist,
    }


def _cost_gap_stats(costs: torch.Tensor) -> dict[str, Any]:
    sorted_costs = torch.sort(costs, dim=1).values
    best = sorted_costs[:, 0]
    second = sorted_costs[:, 1]
    k5_idx = min(4, sorted_costs.shape[1] - 1)
    k10_idx = min(9, sorted_costs.shape[1] - 1)
    gap2 = second - best
    gap5 = sorted_costs[:, k5_idx] - best
    gap10 = sorted_costs[:, k10_idx] - best
    state_std = costs.std(dim=1)
    thresholds = [1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0]
    flat = {}
    for t in thresholds:
        flat[f"second_gap_lt_{t:g}"] = float((gap2 < t).float().mean().item())
        flat[f"top5_margin_lt_{t:g}"] = float((gap5 < t).float().mean().item())
        flat[f"top10_margin_lt_{t:g}"] = float(
            (gap10 < t).float().mean().item()
        )
        flat[f"state_std_lt_{t:g}"] = float(
            (state_std < t).float().mean().item()
        )
    return {
        "best_cost": _stats(best),
        "second_best_cost": _stats(second),
        "second_gap": _stats(gap2),
        "top5_margin": _stats(gap5),
        "top10_margin": _stats(gap10),
        "per_state_std": _stats(state_std),
        "flat_or_ambiguous_fraction": flat,
    }


def _action_distribution(candidate_actions: torch.Tensor) -> dict[str, Any]:
    first_actions = candidate_actions[:, :, 0, :]
    return {
        "all_actions": _stats(candidate_actions),
        "first_actions": _stats(first_actions),
        "action_dim_mean": [
            float(v)
            for v in first_actions.reshape(-1, first_actions.shape[-1]).mean(dim=0)
        ],
        "action_dim_std": [
            float(v)
            for v in first_actions.reshape(-1, first_actions.shape[-1]).std(dim=0)
        ],
    }


def _best_first_action_distribution(
    best_first_action: torch.Tensor,
) -> dict[str, Any]:
    return {
        "global": _stats(best_first_action),
        "dim_mean": [float(v) for v in best_first_action.mean(dim=0)],
        "dim_std": [float(v) for v in best_first_action.std(dim=0)],
    }


def _extract_optional_observation_stats(
    *,
    cache: dict[str, Any],
    env_name: str,
    sample_states: int,
) -> dict[str, Any]:
    keys = ["observation", "proprio", "goal"]
    present = {}
    for k in keys:
        if k in cache and torch.is_tensor(cache[k]):
            present[k] = _stats(cache[k].float())
    if present:
        return {"source": "cache_tensor_keys", "stats": present}

    episodes_idx = list(cache["episodes_idx"])[:sample_states]
    start_steps = list(cache["start_steps"])[:sample_states]
    info = build_info_dict_from_cache(
        env_name=env_name,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset_steps=int(cache["goal_offset_steps"]),
        device="cpu",
    )
    stats: dict[str, Any] = {}
    for k in keys:
        if k in info and torch.is_tensor(info[k]):
            stats[k] = _stats(info[k].float())
    return {
        "source": "rebuilt_info_dict_sample",
        "sample_states": sample_states,
        "stats": stats,
    }


def _cache_block(
    *,
    cache: dict[str, Any],
    env_name: str,
    sample_states: int,
) -> dict[str, Any]:
    costs = cache["teacher_costs"].float()
    candidate_actions = cache["candidate_actions"].float()
    best_idx = cache["teacher_best_index"].long()
    best_first = cache["teacher_best_first_action"].float()
    n_candidates = int(candidate_actions.shape[1])
    return {
        "shape_checks": {
            "candidate_actions": _shape(candidate_actions),
            "teacher_costs": _shape(costs),
            "teacher_best_index": _shape(best_idx),
            "teacher_best_first_action": _shape(best_first),
        },
        "cost_stats": _stats(costs),
        "cost_gap_stats": _cost_gap_stats(costs),
        "teacher_best_index": _best_index_entropy(best_idx, n_candidates),
        "teacher_best_first_action_distribution": _best_first_action_distribution(
            best_first
        ),
        "candidate_action_distribution": _action_distribution(candidate_actions),
        "optional_obs_proprio_goal_stats": _extract_optional_observation_stats(
            cache=cache,
            env_name=env_name,
            sample_states=sample_states,
        ),
    }


def _compare_train_eval(
    train: dict[str, Any],
    eval_: dict[str, Any],
) -> dict[str, Any]:
    train_mean = torch.tensor(
        train["teacher_best_first_action_distribution"]["dim_mean"],
        dtype=torch.float32,
    )
    eval_mean = torch.tensor(
        eval_["teacher_best_first_action_distribution"]["dim_mean"],
        dtype=torch.float32,
    )
    return {
        "candidate_shape_match": (
            train["shape_checks"]["candidate_actions"][1:]
            == eval_["shape_checks"]["candidate_actions"][1:]
        ),
        "cost_mean_delta_eval_minus_train": (
            float(eval_["cost_stats"]["mean"] - train["cost_stats"]["mean"])
        ),
        "cost_std_delta_eval_minus_train": (
            float(eval_["cost_stats"]["std"] - train["cost_stats"]["std"])
        ),
        "second_gap_mean_delta_eval_minus_train": (
            float(
                eval_["cost_gap_stats"]["second_gap"]["mean"]
                - train["cost_gap_stats"]["second_gap"]["mean"]
            )
        ),
        "top10_margin_mean_delta_eval_minus_train": (
            float(
                eval_["cost_gap_stats"]["top10_margin"]["mean"]
                - train["cost_gap_stats"]["top10_margin"]["mean"]
            )
        ),
        "best_index_entropy_delta_eval_minus_train": (
            float(
                eval_["teacher_best_index"]["entropy_bits"]
                - train["teacher_best_index"]["entropy_bits"]
            )
        ),
        "best_first_action_mean_l2_distance": float(
            torch.linalg.norm(train_mean - eval_mean).item()
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    cmp = report["train_eval_comparison"]
    lines = [
        "# Operator Split Diagnostics (TwoRoom)",
        "",
        "## Quick Comparison",
        "",
        f"- Train cache path: `{report['train_cache_path']}`",
        f"- Eval cache path: `{report['eval_cache_path']}`",
        (
            "- Candidate shape match (excluding state count): "
            f"`{cmp['candidate_shape_match']}`"
        ),
        (
            f"- Cost mean delta (eval-train): "
            f"`{cmp['cost_mean_delta_eval_minus_train']:.4f}`"
        ),
        (
            f"- Cost std delta (eval-train): "
            f"`{cmp['cost_std_delta_eval_minus_train']:.4f}`"
        ),
        (
            f"- Second-gap mean delta (eval-train): "
            f"`{cmp['second_gap_mean_delta_eval_minus_train']:.4f}`"
        ),
        (
            f"- Top10-margin mean delta (eval-train): "
            f"`{cmp['top10_margin_mean_delta_eval_minus_train']:.4f}`"
        ),
        (
            f"- Best-index entropy delta (eval-train): "
            f"`{cmp['best_index_entropy_delta_eval_minus_train']:.4f}`"
        ),
        (
            f"- Best-first-action mean L2 distance: "
            f"`{cmp['best_first_action_mean_l2_distance']:.4f}`"
        ),
        "",
        "## Train Cache Highlights",
        "",
        (
            f"- Cost mean/std: `{report['train']['cost_stats']['mean']:.4f}` / "
            f"`{report['train']['cost_stats']['std']:.4f}`"
        ),
        (
            f"- Second-gap mean: "
            f"`{report['train']['cost_gap_stats']['second_gap']['mean']:.4f}`"
        ),
        (
            f"- Top10-margin mean: "
            f"`{report['train']['cost_gap_stats']['top10_margin']['mean']:.4f}`"
        ),
        (
            f"- Best-index entropy (bits): "
            f"`{report['train']['teacher_best_index']['entropy_bits']:.4f}`"
        ),
        "",
        "## Eval Cache Highlights",
        "",
        (
            f"- Cost mean/std: `{report['eval']['cost_stats']['mean']:.4f}` / "
            f"`{report['eval']['cost_stats']['std']:.4f}`"
        ),
        (
            f"- Second-gap mean: "
            f"`{report['eval']['cost_gap_stats']['second_gap']['mean']:.4f}`"
        ),
        (
            f"- Top10-margin mean: "
            f"`{report['eval']['cost_gap_stats']['top10_margin']['mean']:.4f}`"
        ),
        (
            f"- Best-index entropy (bits): "
            f"`{report['eval']['teacher_best_index']['entropy_bits']:.4f}`"
        ),
        "",
        (
            "Detailed distributions and flat/ambiguous fractions are in "
            "`operator_split_diagnostics_tworoom.json`."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument(
        "--train-cache",
        default=(
            "outputs/operator_cache/tworoom/"
            "lewm_tworoom_train_s512_c128_seed0/operator_cache.pt"
        ),
    )
    parser.add_argument(
        "--eval-cache",
        default=(
            "outputs/operator_cache/tworoom/"
            "lewm_tworoom_eval_s128_c128_seed1/operator_cache.pt"
        ),
    )
    parser.add_argument("--sample-info-states", type=int, default=64)
    args = parser.parse_args()

    train_path = Path(args.train_cache)
    eval_path = Path(args.eval_cache)
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train cache: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval cache: {eval_path}")

    train_cache = torch.load(
        train_path,
        map_location="cpu",
        weights_only=False,
    )
    eval_cache = torch.load(
        eval_path,
        map_location="cpu",
        weights_only=False,
    )
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_block = _cache_block(
        cache=train_cache,
        env_name=args.env,
        sample_states=int(args.sample_info_states),
    )
    eval_block = _cache_block(
        cache=eval_cache,
        env_name=args.env,
        sample_states=int(args.sample_info_states),
    )

    report = {
        "env": args.env,
        "train_cache_path": str(train_path),
        "eval_cache_path": str(eval_path),
        "train": train_block,
        "eval": eval_block,
        "train_eval_comparison": _compare_train_eval(train_block, eval_block),
    }

    json_path = out_dir / "operator_split_diagnostics_tworoom.json"
    md_path = out_dir / "operator_split_diagnostics_tworoom.md"
    save_json(json_path, report)
    md_path.write_text(_markdown(report), encoding="utf-8")
    print("Operator split diagnostics written")
    print(f"  json: {json_path}")
    print(f"  md:   {md_path}")


if __name__ == "__main__":
    main()
