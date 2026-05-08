from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from oawc.benchmark import load_hdf5_dataset, sample_dataset_eval_tasks
from oawc.compression.operator_metrics import (
    adapt_candidates_for_model,
    build_info_dict_from_cache,
    compute_model_costs,
    maybe_align_action_width,
    resolve_device,
)
from oawc.envs import ENV_SPECS
from oawc.models import load_cost_model


def _episode_column(dataset: Any) -> str:
    if "episode_idx" in dataset.column_names:
        return "episode_idx"
    if "ep_idx" in dataset.column_names:
        return "ep_idx"
    raise KeyError(
        "Dataset missing episode index column. "
        f"Available columns: {dataset.column_names}"
    )


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _slice_info_dict(
    info: dict[str, torch.Tensor],
    state_slice: slice,
    device: str,
) -> dict[str, torch.Tensor]:
    return {k: v[state_slice].to(device) for k, v in info.items()}


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
            expanded = {
                k: v.unsqueeze(1).repeat(
                    1,
                    int(c1 - c0),
                    *([1] * (v.ndim - 1)),
                )
                for k, v in info_chunk.items()
            }
            with torch.no_grad():
                out = compute_model_costs(model, expanded, cand_eval)
            costs_cpu[s0:s1, c0:c1] = out.detach().to("cpu")
    return costs_cpu


def _sample_dataset_action_candidates(
    *,
    dataset: Any,
    num_states: int,
    num_candidates: int,
    horizon: int,
    seed: int,
    action_noise_std: float,
) -> torch.Tensor:
    ep_col = _episode_column(dataset)
    episode_per_row = dataset.get_col_data(ep_col).astype(np.int64)
    step_per_row = dataset.get_col_data("step_idx").astype(np.int64)
    unique_eps = np.unique(episode_per_row)
    max_step_by_ep = {}
    for ep in unique_eps:
        max_step_by_ep[int(ep)] = int(
            step_per_row[episode_per_row == ep].max()
        )
    valid_mask = np.asarray(
        [
            int(step) + horizon - 1 <= max_step_by_ep[int(ep)]
            for ep, step in zip(episode_per_row, step_per_row)
        ],
        dtype=bool,
    )
    valid_rows = np.nonzero(valid_mask)[0]
    if len(valid_rows) == 0:
        raise ValueError(
            "No valid dataset action windows for requested horizon."
        )

    rng = np.random.default_rng(seed)
    total = int(num_states * num_candidates)
    sampled_rows = rng.choice(valid_rows, size=total, replace=True)
    ep_idx = episode_per_row[sampled_rows]
    start_steps = step_per_row[sampled_rows]

    seqs = dataset.load_chunk(
        ep_idx,
        start_steps,
        start_steps + int(horizon),
    )
    actions = []
    for seq in seqs:
        action_arr = seq["action"]
        if torch.is_tensor(action_arr):
            action_arr = action_arr.detach().cpu().numpy()
        action_arr = np.asarray(action_arr, dtype=np.float32)
        if action_arr.ndim != 2:
            action_arr = action_arr.reshape(action_arr.shape[0], -1)
        if action_arr.shape[0] < horizon:
            raise ValueError(
                "Short action window encountered in dataset chunk."
            )
        actions.append(action_arr[:horizon])
    act_np = np.stack(actions, axis=0).reshape(
        num_states,
        num_candidates,
        horizon,
        -1,
    )
    if action_noise_std > 0:
        noise = rng.normal(
            loc=0.0,
            scale=float(action_noise_std),
            size=act_np.shape,
        ).astype(np.float32)
        act_np = np.clip(act_np + noise, -1.0, 1.0)
    return torch.as_tensor(act_np, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument("--model-family", default="lewm_hf")
    parser.add_argument("--checkpoint", default="quentinll/lewm-tworooms")
    parser.add_argument("--num-states", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--candidate-mode", default="dataset_actions")
    parser.add_argument("--action-noise-std", type=float, default=0.0)
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-candidates", type=int, default=128)
    args = parser.parse_args()

    device = resolve_device(args.device)
    out_dir = Path("outputs/operator_cache") / args.env / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "operator_cache.pt"
    meta_path = out_dir / "metadata.json"

    dataset = load_hdf5_dataset(args.env)
    tasks = sample_dataset_eval_tasks(
        dataset=dataset,
        goal_offset_steps=ENV_SPECS[args.env].goal_distance_steps,
        num_eval=int(args.num_states),
        seed=int(args.seed),
    )

    loaded = load_cost_model(
        family=args.model_family,
        checkpoint=args.checkpoint,
        env_name=args.env,
        device=device,
    )
    teacher = loaded.model.to(device).eval()
    teacher.requires_grad_(False)

    info_cpu = build_info_dict_from_cache(
        env_name=args.env,
        episodes_idx=list(tasks["episodes_idx"]),
        start_steps=list(tasks["start_steps"]),
        goal_offset_steps=int(tasks["goal_offset_steps"]),
        device="cpu",
    )
    info_cpu = maybe_align_action_width(info_cpu, teacher)
    candidate_actions = _sample_dataset_action_candidates(
        dataset=dataset,
        num_states=int(args.num_states),
        num_candidates=int(args.num_candidates),
        horizon=int(args.horizon),
        seed=int(args.seed),
        action_noise_std=float(args.action_noise_std),
    )
    teacher_costs = _compute_costs_chunked(
        model=teacher,
        info_cpu=info_cpu,
        candidate_actions_cpu=candidate_actions,
        device=device,
        batch_states=max(1, int(args.batch_states)),
        batch_candidates=max(1, int(args.batch_candidates)),
    )

    sorted_idx = torch.argsort(teacher_costs, dim=1)
    topk_indices = {
        str(k): sorted_idx[:, : int(min(k, teacher_costs.shape[1]))]
        for k in sorted(int(k) for k in args.topk)
    }
    teacher_best_index = sorted_idx[:, 0]
    teacher_best_first_action = candidate_actions[
        torch.arange(int(args.num_states)),
        teacher_best_index,
        0,
        :,
    ]

    cache = {
        "env": args.env,
        "model_family": args.model_family,
        "checkpoint": args.checkpoint,
        "resolved_checkpoint": args.checkpoint,
        "tag": args.tag,
        "split": "eval",
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "num_states": int(args.num_states),
        "num_candidates": int(args.num_candidates),
        "action_dim": int(candidate_actions.shape[-1]),
        "topk": [int(k) for k in args.topk],
        "episodes_idx": list(tasks["episodes_idx"]),
        "start_steps": list(tasks["start_steps"]),
        "goal_offset_steps": int(tasks["goal_offset_steps"]),
        "candidate_mode": str(args.candidate_mode),
        "candidate_actions": candidate_actions.cpu(),
        "teacher_costs": teacher_costs.cpu(),
        "topk_indices": topk_indices,
        "teacher_best_index": teacher_best_index.cpu(),
        "teacher_best_first_action": teacher_best_first_action.cpu(),
        "teacher_cost_stats": {
            "min": float(torch.min(teacher_costs)),
            "mean": float(torch.mean(teacher_costs)),
            "max": float(torch.max(teacher_costs)),
            "std": float(torch.std(teacher_costs)),
            "finite": bool(torch.isfinite(teacher_costs).all().item()),
        },
    }
    torch.save(cache, cache_path)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "cache_path": str(cache_path),
        "env": args.env,
        "candidate_mode": str(args.candidate_mode),
        "action_noise_std": float(args.action_noise_std),
        "device": device,
        "shape_summary": {
            "candidate_actions": list(candidate_actions.shape),
            "teacher_costs": list(teacher_costs.shape),
        },
    }
    meta_path.write_text(json.dumps(_to_jsonable(meta), indent=2) + "\n")
    print(f"[done] wrote cache: {cache_path}")
    print(f"[done] wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
