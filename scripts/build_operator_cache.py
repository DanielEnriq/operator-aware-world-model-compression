from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F

from oawc.benchmark import (
    count_parameters,
    fit_world_policy_processors,
    image_transform,
    load_hdf5_dataset,
    model_size_bytes,
    sample_dataset_eval_tasks,
)
from oawc.envs import ENV_SPECS
from oawc.models import load_cost_model


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _extract_init_goal(dataset, episodes_idx, start_steps, goal_offset):
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

            # Stable WM datasets usually store pixels in CHW; convert to HWC.
            if col == "pixels" and arr.ndim >= 4:
                arr = np.transpose(arr, (0, 2, 3, 1))

            init_lists.setdefault(col, []).append(arr[0])
            goal_lists.setdefault(col, []).append(arr[-1])

    init_state = {k: np.stack(v) for k, v in init_lists.items()}
    goal_state = {}
    for k, v in goal_lists.items():
        goal_state["goal" if k == "pixels" else f"goal_{k}"] = np.stack(v)

    return init_state, goal_state


def _ensure_lane_import(project_root: Path, source_swm: bool) -> Path:
    swm_file = Path(swm.__file__).resolve()
    source_root = (project_root / "external" / "stable-worldmodel").resolve()
    in_source_lane = source_root in swm_file.parents
    if source_swm and not in_source_lane:
        raise RuntimeError(
            "Expected source-SWM lane import from external/stable-worldmodel, "
            f"got: {swm_file}"
        )
    if not source_swm and in_source_lane:
        raise RuntimeError(
            "Expected installed-SWM lane (not external/stable-worldmodel), "
            f"got: {swm_file}"
        )
    return swm_file


def _resolve_object_checkpoint(checkpoint: str, stablewm_home: Path) -> Path:
    ckpt_input = Path(checkpoint).expanduser()
    if not ckpt_input.is_absolute():
        ckpt_input = (Path.cwd() / ckpt_input).resolve()

    if ckpt_input.is_file():
        return ckpt_input

    run_dir_candidates: list[Path] = []
    if ckpt_input.is_dir():
        run_dir_candidates.append(ckpt_input)
    run_dir_candidates.append(stablewm_home / "checkpoints" / checkpoint)

    for run_dir in run_dir_candidates:
        if not run_dir.exists() or not run_dir.is_dir():
            continue
        preferred = run_dir / f"{run_dir.name}_object.ckpt"
        if preferred.exists():
            return preferred
        object_ckpts = sorted(run_dir.glob("*_object.ckpt"))
        if object_ckpts:
            return object_ckpts[0]
    raise FileNotFoundError(
        "Could not resolve object checkpoint for "
        f"'{checkpoint}' in {stablewm_home}"
    )


def _load_source_swm_model(
    checkpoint: str,
    stablewm_home: Path,
) -> tuple[Any, str]:
    object_ckpt = _resolve_object_checkpoint(checkpoint, stablewm_home)
    loaded = torch.load(object_ckpt, map_location="cpu", weights_only=False)
    model = getattr(loaded, "model", loaded)
    return model, str(object_ckpt)


def _build_info_dict(
    *,
    dataset,
    episodes_idx: list[int],
    start_steps: list[int],
    goal_offset_steps: int,
    device: str,
) -> dict[str, torch.Tensor]:
    init_state, goal_state = _extract_init_goal(
        dataset,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset=goal_offset_steps,
    )

    info_np = {}
    info_np.update(init_state)
    info_np.update(goal_state)
    info_np["id"] = np.asarray(episodes_idx, dtype=np.int64)
    info_np["step_idx"] = np.asarray(start_steps, dtype=np.int64)

    # Match policy preprocessors used in official benchmarks.
    process = fit_world_policy_processors(
        dataset,
        keys_to_process=["action", "proprio"],
    )

    for key, scaler in process.items():
        if key not in info_np:
            continue
        arr = info_np[key]
        if arr.ndim == 1:
            arr_2d = arr[:, None]
        else:
            arr_2d = arr.reshape(arr.shape[0], -1)
        info_np[key] = scaler.transform(arr_2d)

    img_tf = image_transform(224)

    def transform_images(arr: np.ndarray) -> torch.Tensor:
        # Input expected as [B, H, W, C] uint8/float; output [B, 1, C, H, W].
        out = [img_tf(img) for img in arr]
        return torch.stack(out, dim=0).unsqueeze(1)

    if "pixels" in info_np:
        info_pixels = info_np["pixels"]
        info_np["pixels"] = transform_images(info_pixels)
    if "goal" in info_np:
        info_goal = info_np["goal"]
        info_np["goal"] = transform_images(info_goal)

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


def _sample_candidates(
    *,
    num_states: int,
    num_candidates: int,
    horizon: int,
    action_dim: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    # v0: sample uniformly in [-1, 1].
    candidates = (
        torch.rand(
            (num_states, num_candidates, horizon, action_dim),
            generator=gen,
            dtype=torch.float32,
        )
        * 2.0
        - 1.0
    )
    return candidates.to(device)


def _expand_info_for_candidates(
    info_dict: dict[str, torch.Tensor],
    num_candidates: int,
) -> dict[str, torch.Tensor]:
    expanded: dict[str, torch.Tensor] = {}
    for key, value in info_dict.items():
        repeats = [1, num_candidates] + [1] * (value.ndim - 1)
        expanded[key] = value.unsqueeze(1).repeat(*repeats)
    return expanded


def _maybe_align_action_width(
    info_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    action = info_dict.get("action")
    if action is None:
        return info_dict

    expected = _get_model_action_width(model)
    if expected is None:
        return info_dict

    current = int(action.shape[-1])
    if current == expected:
        return info_dict
    if expected % current != 0:
        raise ValueError(
            f"Action width mismatch: model expects {expected}, got {current}."
        )

    repeat_factor = expected // current
    info_dict["action"] = action.repeat_interleave(repeat_factor, dim=-1)
    return info_dict


def _get_model_action_width(model: torch.nn.Module) -> int | None:
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


def _adapt_candidates_for_model(
    raw_candidates: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    expected = _get_model_action_width(model)
    if expected is None:
        return raw_candidates

    current = int(raw_candidates.shape[-1])
    if current == expected:
        return raw_candidates
    if expected % current != 0:
        raise ValueError(
            "Candidate action width mismatch: "
            f"expected {expected}, got {current}."
        )
    return raw_candidates.repeat_interleave(expected // current, dim=-1)


def _compute_cost_with_jepa_fallback(
    model: torch.nn.Module,
    info_dict: dict[str, torch.Tensor],
    action_candidates: torch.Tensor,
) -> torch.Tensor:
    # This mirrors JEPA.get_cost but handles goal embedding expansion robustly
    # for models that produce goal_emb with shape [B, T, D].
    goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
    goal["pixels"] = goal["goal"]
    for key in list(goal.keys()):
        if key.startswith("goal_"):
            goal[key[len("goal_"):]] = goal.pop(key)
    goal.pop("action", None)
    goal = model.encode(goal)
    goal_emb = goal["emb"]

    rollout_input = dict(info_dict)
    rollout_input["goal_emb"] = goal_emb
    rollout_output = model.rollout(rollout_input, action_candidates)
    pred_emb = rollout_output["predicted_emb"]

    # Expand goal embeddings across samples/time when needed.
    while goal_emb.ndim < pred_emb.ndim:
        goal_emb = goal_emb.unsqueeze(1)
    goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

    return F.mse_loss(
        pred_emb[..., -1:, :],
        goal_emb[..., -1:, :].detach(),
        reduction="none",
    ).sum(dim=tuple(range(2, pred_emb.ndim)))


def _compute_teacher_costs(
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


def _compute_teacher_costs_chunked(
    *,
    model: torch.nn.Module,
    info_dict_cpu: dict[str, torch.Tensor],
    candidate_actions_cpu: torch.Tensor,
    cost_batch_states: int,
    cost_batch_candidates: int,
    empty_cache_between_batches: bool,
    device: str,
) -> torch.Tensor:
    num_states = int(candidate_actions_cpu.shape[0])
    num_candidates = int(candidate_actions_cpu.shape[1])
    teacher_costs_cpu = torch.empty(
        (num_states, num_candidates),
        dtype=torch.float32,
        device="cpu",
    )

    state_bs = max(1, int(cost_batch_states))
    cand_bs = max(1, int(cost_batch_candidates))
    state_start = 0

    while state_start < num_states:
        curr_state_bs = min(state_bs, num_states - state_start)
        state_slice = slice(state_start, state_start + curr_state_bs)

        try:
            cand_start = 0
            while cand_start < num_candidates:
                curr_cand_bs = min(cand_bs, num_candidates - cand_start)
                cand_slice = slice(cand_start, cand_start + curr_cand_bs)

                if torch.cuda.is_available() and device == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                info_chunk = _slice_info_dict(info_dict_cpu, state_slice, device)
                candidate_chunk = candidate_actions_cpu[state_slice, cand_slice].to(
                    device
                )
                candidate_eval = _adapt_candidates_for_model(candidate_chunk, model)
                expanded_info = _expand_info_for_candidates(
                    info_chunk,
                    num_candidates=curr_cand_bs,
                )
                costs_chunk = _compute_teacher_costs(
                    model,
                    expanded_info,
                    candidate_eval,
                )
                teacher_costs_cpu[state_slice, cand_slice] = (
                    costs_chunk.detach().to("cpu")
                )

                info_shapes = {k: list(v.shape) for k, v in info_chunk.items()}
                peak_gib = _cuda_peak_gib()
                print(
                    "[cost-chunk] "
                    f"states={state_slice.start}:{state_slice.stop} "
                    f"cands={cand_slice.start}:{cand_slice.stop} "
                    f"candidate_shape={list(candidate_eval.shape)} "
                    f"cost_shape={list(costs_chunk.shape)} "
                    f"cuda_peak_gib={peak_gib}"
                )
                print(f"[cost-chunk] info_shapes={info_shapes}")

                del info_chunk
                del candidate_chunk
                del candidate_eval
                del expanded_info
                del costs_chunk
                if (
                    empty_cache_between_batches
                    and torch.cuda.is_available()
                    and device == "cuda"
                ):
                    torch.cuda.empty_cache()
                cand_start += curr_cand_bs

            state_start += curr_state_bs
        except RuntimeError as exc:
            msg = str(exc).lower()
            is_oom = (
                "out of memory" in msg
                or "cuda out of memory" in msg
                or "torch.outofmemoryerror" in msg
            )
            if not is_oom:
                raise
            if torch.cuda.is_available() and device == "cuda":
                torch.cuda.empty_cache()
            next_sizes = _oom_retry_batch_sizes(state_bs, cand_bs)
            if next_sizes is None:
                raise RuntimeError(
                    "OOM persisted at cost_batch_states=1 and "
                    "cost_batch_candidates=1. Cannot proceed."
                ) from exc
            state_bs, cand_bs = next_sizes
            print(
                "[oom-retry] lowering chunk sizes and retrying: "
                f"cost_batch_states={state_bs}, "
                f"cost_batch_candidates={cand_bs}"
            )

    return teacher_costs_cpu


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. "
            "Use --device cpu locally or run on Colab/H100."
        )
    return device_arg


def _slice_info_dict(
    info_dict: dict[str, torch.Tensor],
    state_slice: slice,
    device: str,
) -> dict[str, torch.Tensor]:
    sliced: dict[str, torch.Tensor] = {}
    for key, value in info_dict.items():
        item = value[state_slice]
        sliced[key] = item.to(device)
    return sliced


def _oom_retry_batch_sizes(
    state_bs: int,
    cand_bs: int,
) -> tuple[int, int] | None:
    if state_bs > 1:
        return max(1, state_bs // 2), cand_bs
    if cand_bs > 1:
        return state_bs, max(1, cand_bs // 2)
    return None


def _cuda_peak_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024**3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=list(ENV_SPECS.keys()))
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-states", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-swm", action="store_true")
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "eval"],
        help="Optional cache split label stored in metadata/cache.",
    )
    parser.add_argument("--cost-batch-states", type=int, default=16)
    parser.add_argument("--cost-batch-candidates", type=int, default=128)
    parser.add_argument(
        "--empty-cache-between-cost-batches",
        type=_str2bool,
        default=True,
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    stablewm_home = Path(
        os.environ.get("STABLEWM_HOME", str(project_root / ".swm_cache"))
    ).expanduser()
    stablewm_home.mkdir(parents=True, exist_ok=True)

    out_dir = Path("outputs/operator_cache") / args.env / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "operator_cache.pt"
    metadata_path = out_dir / "metadata.json"

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "env": args.env,
        "model_family": args.model_family,
        "checkpoint": args.checkpoint,
        "tag": args.tag,
        "num_states": args.num_states,
        "num_candidates": args.num_candidates,
        "horizon": args.horizon,
        "topk": args.topk,
        "seed": args.seed,
        "source_swm": bool(args.source_swm),
        "split": args.split,
        "cost_batch_states": int(args.cost_batch_states),
        "cost_batch_candidates": int(args.cost_batch_candidates),
        "empty_cache_between_cost_batches": bool(
            args.empty_cache_between_cost_batches
        ),
        "stablewm_home": str(stablewm_home),
        "project_root": str(project_root),
    }

    try:
        swm_file = _ensure_lane_import(
            project_root,
            source_swm=args.source_swm,
        )
        metadata["stable_worldmodel_file"] = str(swm_file)

        device = _resolve_device(args.device)
        metadata["device"] = device

        print("[progress] loading dataset")
        dataset = load_hdf5_dataset(args.env)
        tasks = sample_dataset_eval_tasks(
            dataset=dataset,
            goal_offset_steps=ENV_SPECS[args.env].goal_distance_steps,
            num_eval=args.num_states,
            seed=args.seed,
        )
        episodes_idx = tasks["episodes_idx"]
        start_steps = tasks["start_steps"]
        goal_offset_steps = tasks["goal_offset_steps"]

        print("[progress] loading teacher model")
        resolved_checkpoint = args.checkpoint
        if args.source_swm:
            if args.model_family != "swm_auto":
                raise ValueError(
                    "source-SWM mode currently supports "
                    "--model-family swm_auto."
                )
            model, resolved_checkpoint = _load_source_swm_model(
                args.checkpoint,
                stablewm_home,
            )
            model = model.to(device).eval()
            model.requires_grad_(False)
        else:
            loaded = load_cost_model(
                family=args.model_family,
                checkpoint=args.checkpoint,
                env_name=args.env,
                device=device,
            )
            model = loaded.model

        if not hasattr(model, "get_cost"):
            raise TypeError("Loaded teacher model does not expose get_cost().")

        print("[progress] preparing state/goal info dict")
        info_dict_cpu = _build_info_dict(
            dataset=dataset,
            episodes_idx=episodes_idx,
            start_steps=start_steps,
            goal_offset_steps=goal_offset_steps,
            device="cpu",
        )
        info_dict_cpu = _maybe_align_action_width(info_dict_cpu, model)

        action_dim = ENV_SPECS[args.env].action_dim or int(
            info_dict_cpu["action"].shape[-1]
        )
        metadata["action_dim"] = action_dim

        print("[progress] sampling candidate action sequences")
        candidate_actions_raw = _sample_candidates(
            num_states=args.num_states,
            num_candidates=args.num_candidates,
            horizon=args.horizon,
            action_dim=action_dim,
            seed=args.seed,
            device="cpu",
        )

        print("[progress] evaluating teacher operator costs")
        with torch.no_grad():
            teacher_costs = _compute_teacher_costs_chunked(
                model=model,
                info_dict_cpu=info_dict_cpu,
                candidate_actions_cpu=candidate_actions_raw,
                cost_batch_states=args.cost_batch_states,
                cost_batch_candidates=args.cost_batch_candidates,
                empty_cache_between_batches=(
                    args.empty_cache_between_cost_batches
                ),
                device=device,
            )
        candidate_actions_cpu = candidate_actions_raw

        max_topk = max(args.topk)
        sorted_idx = torch.argsort(teacher_costs, dim=1)
        topk_indices = {
            str(k): sorted_idx[:, :k]
            for k in args.topk
            if k <= max_topk and k <= args.num_candidates
        }
        teacher_best_index = sorted_idx[:, 0]
        teacher_best_first_action = candidate_actions_cpu[
            torch.arange(args.num_states), teacher_best_index, 0, :
        ]

        teacher_cost_stats = {
            "min": float(torch.min(teacher_costs)),
            "mean": float(torch.mean(teacher_costs)),
            "max": float(torch.max(teacher_costs)),
            "std": float(torch.std(teacher_costs)),
            "finite": bool(torch.isfinite(teacher_costs).all().item()),
        }

        cache = {
            "env": args.env,
            "model_family": args.model_family,
            "checkpoint": args.checkpoint,
            "resolved_checkpoint": resolved_checkpoint,
            "tag": args.tag,
            "split": args.split,
            "seed": args.seed,
            "horizon": args.horizon,
            "num_states": args.num_states,
            "num_candidates": args.num_candidates,
            "action_dim": action_dim,
            "topk": args.topk,
            "episodes_idx": episodes_idx,
            "start_steps": start_steps,
            "goal_offset_steps": goal_offset_steps,
            "candidate_actions": candidate_actions_cpu,
            "teacher_costs": teacher_costs,
            "topk_indices": topk_indices,
            "teacher_best_index": teacher_best_index,
            "teacher_best_first_action": teacher_best_first_action,
            "teacher_cost_stats": teacher_cost_stats,
            "model_size_bytes": model_size_bytes(model),
            "parameter_count": count_parameters(model),
        }
        torch.save(cache, cache_path)

        metadata.update(
            {
                "status": "success",
                "resolved_checkpoint": resolved_checkpoint,
                "cache_path": str(cache_path),
                "shape_summary": {
                    "candidate_actions": list(candidate_actions_cpu.shape),
                    "teacher_costs": list(teacher_costs.shape),
                    "topk_indices": {
                        k: list(v.shape) for k, v in topk_indices.items()
                    },
                    "teacher_best_index": list(teacher_best_index.shape),
                    "teacher_best_first_action": list(
                        teacher_best_first_action.shape
                    ),
                },
                "teacher_cost_stats": teacher_cost_stats,
                "model_size_bytes": int(model_size_bytes(model)),
                "parameter_count": int(count_parameters(model)),
            }
        )
        print(f"[done] wrote cache: {cache_path}")
    except Exception as exc:
        metadata.update(
            {
                "status": "failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(metadata), f, indent=2)
        raise
    else:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(metadata), f, indent=2)
        print(f"[done] wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
