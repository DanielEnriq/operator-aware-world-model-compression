from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from oawc.compression.operator_metrics import (
    adapt_candidates_for_model,
    build_info_dict_from_cache,
    compute_model_costs,
    maybe_align_action_width,
    resolve_device,
)
from oawc.models import load_cost_model


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def _slice_info_dict(
    info: dict[str, torch.Tensor],
    state_slice: slice,
    device: str,
) -> dict[str, torch.Tensor]:
    return {k: v[state_slice].to(device) for k, v in info.items()}


def _expand_info(
    info: dict[str, torch.Tensor],
    num_candidates: int,
) -> dict[str, torch.Tensor]:
    return {
        k: v.unsqueeze(1).repeat(1, num_candidates, *([1] * (v.ndim - 1)))
        for k, v in info.items()
    }


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
    out = torch.empty((num_states, num_candidates), dtype=torch.float32)
    for s0 in range(0, num_states, batch_states):
        s1 = min(num_states, s0 + batch_states)
        info_chunk = _slice_info_dict(info_cpu, slice(s0, s1), device)
        for c0 in range(0, num_candidates, batch_candidates):
            c1 = min(num_candidates, c0 + batch_candidates)
            cand_chunk = candidate_actions_cpu[s0:s1, c0:c1].to(device)
            cand_eval = adapt_candidates_for_model(cand_chunk, model)
            with torch.no_grad():
                costs = compute_model_costs(
                    model,
                    _expand_info(info_chunk, int(c1 - c0)),
                    cand_eval,
                )
            out[s0:s1, c0:c1] = costs.detach().cpu()
    return out


def _build_near_elite_candidates(
    *,
    candidate_actions: torch.Tensor,
    teacher_costs: torch.Tensor,
    num_candidates: int,
    elite_k: int,
    noise_std: float,
    seed: int,
) -> torch.Tensor:
    n_states = int(candidate_actions.shape[0])
    k = max(1, min(int(elite_k), int(candidate_actions.shape[1])))
    sorted_idx = torch.argsort(teacher_costs, dim=1)
    elite_idx = sorted_idx[:, :k]
    elite_actions = candidate_actions[
        torch.arange(n_states).unsqueeze(1),
        elite_idx,
    ]  # [N, k, H, A]
    if int(num_candidates) <= k:
        return elite_actions[:, :num_candidates].clone()

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    choice = torch.randint(
        low=0,
        high=k,
        size=(n_states, int(num_candidates - k)),
        generator=gen,
    )
    sampled = elite_actions[
        torch.arange(n_states).unsqueeze(1),
        choice,
    ]
    if float(noise_std) > 0:
        noise = torch.randn_like(sampled, generator=gen) * float(noise_std)
        sampled = (sampled + noise).clamp(-1.0, 1.0)
    merged = torch.cat([elite_actions, sampled], dim=1)
    return merged[:, :num_candidates].contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--elite-k", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-candidates", type=int, default=128)
    args = parser.parse_args()

    source_path = Path(args.source_cache)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    env = str(source["env"])
    out_dir = Path("outputs/operator_cache") / env / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "operator_cache.pt"
    meta_path = out_dir / "metadata.json"

    candidate_actions_src = source["candidate_actions"].float()
    teacher_costs_src = source["teacher_costs"].float()
    near_elite = _build_near_elite_candidates(
        candidate_actions=candidate_actions_src,
        teacher_costs=teacher_costs_src,
        num_candidates=int(args.num_candidates),
        elite_k=int(args.elite_k),
        noise_std=float(args.noise_std),
        seed=int(args.seed),
    )

    device = resolve_device(args.device)
    loaded = load_cost_model(
        family=str(source["model_family"]),
        checkpoint=str(
            source.get("resolved_checkpoint", source["checkpoint"])
        ),
        env_name=env,
        device=device,
    )
    teacher = loaded.model.to(device).eval()
    teacher.requires_grad_(False)
    info_cpu = build_info_dict_from_cache(
        env_name=env,
        episodes_idx=list(source["episodes_idx"]),
        start_steps=list(source["start_steps"]),
        goal_offset_steps=int(source["goal_offset_steps"]),
        device="cpu",
    )
    info_cpu = maybe_align_action_width(info_cpu, teacher)
    teacher_costs = _compute_costs_chunked(
        model=teacher,
        info_cpu=info_cpu,
        candidate_actions_cpu=near_elite,
        device=device,
        batch_states=max(1, int(args.batch_states)),
        batch_candidates=max(1, int(args.batch_candidates)),
    )

    sorted_idx = torch.argsort(teacher_costs, dim=1)
    teacher_best_index = sorted_idx[:, 0]
    teacher_best_first_action = near_elite[
        torch.arange(near_elite.shape[0]),
        teacher_best_index,
        0,
        :,
    ]
    topk = [int(k) for k in source.get("topk", [1, 5, 10, 20])]
    topk_indices = {
        str(k): sorted_idx[:, : int(min(k, teacher_costs.shape[1]))]
        for k in sorted(topk)
    }
    cache = {
        **source,
        "tag": args.tag,
        "seed": int(args.seed),
        "candidate_mode": "near_elite",
        "source_cache": str(source_path),
        "num_candidates": int(near_elite.shape[1]),
        "candidate_actions": near_elite.cpu(),
        "teacher_costs": teacher_costs.cpu(),
        "teacher_best_index": teacher_best_index.cpu(),
        "teacher_best_first_action": teacher_best_first_action.cpu(),
        "topk_indices": topk_indices,
    }
    torch.save(cache, cache_path)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "cache_path": str(cache_path),
        "source_cache": str(source_path),
        "candidate_mode": "near_elite",
        "elite_k": int(args.elite_k),
        "noise_std": float(args.noise_std),
        "device": device,
        "shape_summary": {
            "candidate_actions": list(near_elite.shape),
            "teacher_costs": list(teacher_costs.shape),
        },
    }
    meta_path.write_text(json.dumps(_to_jsonable(meta), indent=2) + "\n")
    print(f"[done] wrote cache: {cache_path}")
    print(f"[done] wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
