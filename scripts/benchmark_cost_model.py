from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import stable_pretraining as spt
import stable_worldmodel as swm
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from oawc.models import (
    available_model_families,
    load_cost_model as load_registered_cost_model,
)

from oawc.benchmark import (
    get_dataset_name,
    load_hdf5_dataset,
    sample_dataset_eval_tasks,
)
from oawc.envs import (
    ENV_SPECS,
    get_cem_solver_kwargs,
    get_eval_callables,
    get_planning_config_kwargs,
    make_world,
)


def ensure_lewm_source_on_path() -> None:
    """
    Required for torch-loaded LeWM objects whose classes live in external/le-wm.

    This does not change the benchmark protocol. It only makes deserialization
    robust for local compressed models saved from the LeWM codebase.
    """
    project_root = Path(__file__).resolve().parents[1]
    lewm_src = project_root / "external" / "le-wm"

    if lewm_src.exists() and str(lewm_src) not in sys.path:
        sys.path.insert(0, str(lewm_src))


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _model_report_metadata(model_path: str | None) -> dict[str, Any] | None:
    if model_path is None:
        return None
    p = Path(model_path)
    parent = p.parent
    compression = _load_optional_json(parent / "compression_report.json")
    distill = _load_optional_json(parent / "distillation_report.json")
    return {
        "compression_report": compression,
        "distillation_report": distill,
        "compression_report_path": (
            str(parent / "compression_report.json")
            if (parent / "compression_report.json").exists()
            else None
        ),
        "distillation_report_path": (
            str(parent / "distillation_report.json")
            if (parent / "distillation_report.json").exists()
            else None
        ),
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _extract_metric(
    raw_metrics: dict[str, Any],
    keys: list[str],
) -> float | None:
    for key in keys:
        if key not in raw_metrics:
            continue
        value = raw_metrics[key]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for field in ("mean", "avg", "value"):
                field_value = value.get(field)
                if isinstance(field_value, (int, float)):
                    return float(field_value)
            numeric_values = [
                float(v)
                for v in value.values()
                if isinstance(v, (int, float))
            ]
            maybe_mean = _mean_or_none(numeric_values)
            if maybe_mean is not None:
                return maybe_mean
        if isinstance(value, list):
            numeric_values = [
                float(v) for v in value if isinstance(v, (int, float))
            ]
            maybe_mean = _mean_or_none(numeric_values)
            if maybe_mean is not None:
                return maybe_mean
    return None


def imagenet_transform(img_size: int = 224):
    """
    Same image preprocessing pattern used in external/le-wm/eval.py:
      ToImage -> float in [0,1] -> ImageNet normalization -> Resize.
    """
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def fit_policy_processors(dataset: Any, keys_to_process: list[str]) -> dict[str, Any]:
    """
    Match external/le-wm/eval.py.

    Non-pixel columns are standardized using dataset-wide statistics. For each
    non-action column, we also create a goal_<col> processor because SWM's
    dataset-driven evaluation provides goal fields.
    """
    process: dict[str, Any] = {}

    for col in keys_to_process:
        if col == "pixels":
            continue
        if col not in dataset.column_names:
            continue

        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = processor

    return process


def load_cost_model_legacy(
    *,
    model_family: str,
    checkpoint: str,
    device: str,
) -> torch.nn.Module:
    """
    Load a model implementing SWM's cost-model contract:

        get_cost(info_dict, action_candidates) -> costs

    Supported modes:
      - auto: use swm.policy.AutoCostModel(checkpoint)
      - torch: torch.load(checkpoint), useful for compressed local .pt models
      - lewm_hf: use oawc.models.lewm_loader.load_lewm_from_hf(checkpoint)

    For original SWM-compatible checkpoints, prefer --model-family auto.
    For the HF LeWM checkpoints already used in this repo, use --model-family lewm_hf.
    """
    ensure_lewm_source_on_path()

    if model_family == "auto":
        model = swm.policy.AutoCostModel(checkpoint)

    elif model_family == "torch":
        model = torch.load(checkpoint, map_location=device, weights_only=False)

    elif model_family == "lewm_hf":
        from oawc.models.lewm_loader import load_lewm_from_hf

        model = load_lewm_from_hf(checkpoint, device=device)

    else:
        raise ValueError(
            f"Unknown model_family={model_family}. "
            "Valid: auto, torch, lewm_hf."
        )

    model = model.to(device)
    model = model.eval()
    model.requires_grad_(False)

    # LeWM/DINO-style ViT encoders may need this for arbitrary eval image sizes.
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    return model


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def model_size_bytes(module: torch.nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in module.parameters()))


def _compute_cost_with_fallback(
    model: torch.nn.Module,
    info_dict: dict[str, torch.Tensor],
    action_candidates: torch.Tensor,
) -> torch.Tensor:
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

    while goal_emb.ndim < pred_emb.ndim:
        goal_emb = goal_emb.unsqueeze(1)
    goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
    return F.mse_loss(
        pred_emb[..., -1:, :],
        goal_emb[..., -1:, :].detach(),
        reduction="none",
    ).sum(dim=tuple(range(2, pred_emb.ndim)))


def _configure_cost_interface(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, str]:
    if callable(getattr(model, "get_cost", None)):
        return model, "get_cost"

    if callable(getattr(model, "cost", None)):
        model.get_cost = model.cost  # type: ignore[attr-defined]
        return model, "cost"

    if callable(getattr(model, "forward", None)):
        model.get_cost = model.forward  # type: ignore[attr-defined]
        return model, "forward"

    if callable(getattr(model, "rollout", None)) and callable(
        getattr(model, "encode", None)
    ):
        def _fallback_get_cost(
            info_dict: dict[str, torch.Tensor],
            action_candidates: torch.Tensor,
            _model: torch.nn.Module = model,
        ) -> torch.Tensor:
            return _compute_cost_with_fallback(
                _model,
                info_dict,
                action_candidates,
            )

        model.get_cost = (  # type: ignore[attr-defined]
            _fallback_get_cost
        )
        return model, "fallback"

    available = [
        name
        for name in [
            "get_cost",
            "cost",
            "forward",
            "encode",
            "rollout",
            "predict",
            "step",
        ]
        if callable(getattr(model, name, None))
    ]
    if callable(getattr(model, "get_cost", None)) or callable(
        getattr(model, "cost", None)
    ):
        mode = "cost_model_direct"
    elif callable(getattr(model, "forward", None)) and not (
        callable(getattr(model, "encode", None))
        and callable(getattr(model, "rollout", None))
    ):
        mode = "forward_only"
    elif callable(getattr(model, "encode", None)) and callable(
        getattr(model, "rollout", None)
    ):
        mode = "representation_rollout_only"
    else:
        mode = "no_planning_interface"

    raise TypeError(
        "Loaded model does not expose a planning cost interface. "
        "Expected one of: get_cost, cost, forward, or "
        "encode+rollout fallback. "
        f"Available callables: {available or ['<none>']}. "
        f"Inferred interface_mode={mode}."
    )


def make_world_model_policy(
    *,
    env_name: str,
    model: torch.nn.Module,
    dataset: Any,
    device: str,
    seed: int,
) -> swm.policy.WorldModelPolicy:
    spec = ENV_SPECS[env_name]

    # Match the LeWM/SWM evaluation contract: only normalize columns that the
    # policy/model actually consumes. In particular, do not standardize the raw
    # "observation" field from world.infos; for TwoRoom it is represented as an
    # object array of tensors at runtime and is not used by LeWM's pixel JEPA
    # cost model. LeWM consumes pixels/goal through torchvision transforms and
    # action through the action processor.
    # Match external/le-wm/config/eval/*.yaml.
    # TwoRoom explicitly caches/processes action and proprio; the eval callables
    # use proprio to set the simulator state and goal_proprio to set the goal.
    # We still avoid raw "observation", which is not consumed by the LeWM cost
    # model and is represented inconsistently in live world.infos.
    keys_to_process = ["action", "proprio"]

    process = fit_policy_processors(
        dataset,
        keys_to_process=keys_to_process,
    )

    transform = {
        "pixels": imagenet_transform(spec.image_shape[0]),
        "goal": imagenet_transform(spec.image_shape[0]),
    }

    plan_config = swm.policy.PlanConfig(**get_planning_config_kwargs(env_name))

    solver = swm.solver.CEMSolver(
        model=model,
        **get_cem_solver_kwargs(env_name, device=device, seed=seed),
    )

    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )


def run_cost_model_benchmark(
    *,
    env_name: str,
    model_family: str,
    checkpoint: str,
    tag: str,
    num_eval: int,
    seed: int,
    device: str,
    save_video: bool,
    output_dir: Path,
    model_path: str | None = None,
) -> dict[str, Any]:
    if env_name not in ENV_SPECS:
        raise KeyError(f"Unknown env={env_name}. Valid envs: {sorted(ENV_SPECS)}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    spec = ENV_SPECS[env_name]
    dataset_name = get_dataset_name(env_name)

    dataset = load_hdf5_dataset(env_name)

    eval_tasks = sample_dataset_eval_tasks(
        dataset=dataset,
        goal_offset_steps=spec.goal_distance_steps,
        num_eval=num_eval,
        seed=seed,
    )

    episodes_idx = eval_tasks["episodes_idx"]
    start_steps = eval_tasks["start_steps"]

    loaded_model = load_registered_cost_model(
        family=model_family,
        checkpoint=checkpoint,
        env_name=env_name,
        device=device,
    )
    model, cost_interface = _configure_cost_interface(loaded_model.model)

    policy = make_world_model_policy(
        env_name=env_name,
        model=model,
        dataset=dataset,
        device=device,
        seed=seed,
    )

    world = make_world(
        env_name,
        num_envs=num_eval,
        seed=seed,
        max_episode_steps=2 * spec.eval_budget_steps,
        goal_conditioned=True,
        verbose=0,
    )
    world.set_policy(policy)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "videos"

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    eval_callables = get_eval_callables(env_name)

    metrics = world.evaluate_from_dataset(
        dataset,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset_steps=spec.goal_distance_steps,
        eval_budget=spec.eval_budget_steps,
        callables=eval_callables,
        save_video=save_video,
        video_path=video_path,
    )
    end_time = time.time()

    evaluation_time_sec = end_time - start_time

    cuda_memory_bytes = None
    if device == "cuda":
        cuda_memory_bytes = int(torch.cuda.max_memory_allocated())

    cem_kwargs = get_cem_solver_kwargs(env_name, device=device, seed=seed)
    plan_kwargs = get_planning_config_kwargs(env_name)

    result = {
        "benchmark_version": "oawc_dataset_driven_v1",
        "model": {
            "name": tag,
            "family": model_family,
            "checkpoint": checkpoint,
            "model_path": model_path,
            "loader_source": loaded_model.source,
            "cost_interface": cost_interface,
            "compression": None,
            "is_cost_model": True,
        },
        "environment": {
            "name": env_name,
            "env_id": spec.env_id,
            "image_shape": list(spec.image_shape),
            "history_size": spec.history_size,
            "frame_skip": spec.frame_skip,
            "action_block": spec.action_block,
            "action_dim": spec.action_dim,
            "observation_kind": spec.observation_kind,
        },
        "dataset": {
            "name": dataset_name,
            "num_episodes_total": int(len(dataset.lengths)),
            "num_clips_total": int(len(dataset)),
            "columns": list(dataset.column_names),
        },
        "evaluation_protocol": {
            "type": "dataset_driven_goal_conditioned",
            "num_eval": num_eval,
            "callables": eval_callables,
            "seed": seed,
            "episodes_idx": episodes_idx,
            "start_steps": start_steps,
            "goal_offset_steps": spec.goal_distance_steps,
            "eval_budget_steps": spec.eval_budget_steps,
            "max_episode_steps": 2 * spec.eval_budget_steps,
            "save_video": save_video,
        },
        "planning": {
            "solver": "CEM",
            "cem_samples": cem_kwargs["num_samples"],
            "cem_elites": cem_kwargs["topk"],
            "cem_iterations": cem_kwargs["n_steps"],
            "cem_initial_variance": cem_kwargs["var_scale"],
            "planning_horizon_blocks": plan_kwargs["horizon"],
            "receding_horizon_blocks": plan_kwargs["receding_horizon"],
            "warm_start": plan_kwargs["warm_start"],
        },
        "performance": {
            "success_rate": metrics.get("success_rate"),
            "episode_successes": metrics.get("episode_successes"),
            "num_successes": (
                int(sum(int(bool(x)) for x in metrics.get("episode_successes", [])))
                if isinstance(metrics.get("episode_successes"), list)
                else None
            ),
            "raw_metrics": metrics,
            "avg_return": _extract_metric(
                metrics,
                [
                    "avg_return",
                    "return",
                    "returns",
                    "episode_return",
                    "episode_returns",
                    "mean_return",
                ],
            ),
            "avg_final_distance": _extract_metric(
                metrics,
                [
                    "avg_final_distance",
                    "final_distance",
                    "distance_to_goal",
                    "goal_distance",
                    "episode_final_distance",
                ],
            ),
        },
        "efficiency": {
            "evaluation_time_sec": evaluation_time_sec,
            "episodes_per_sec": num_eval / evaluation_time_sec if evaluation_time_sec > 0 else None,
            "model_parameters": count_parameters(model),
            "model_size_bytes": model_size_bytes(model),
            "cost_model_latency_sec": None,
            "candidate_throughput_per_sec": None,
            "cuda_memory_bytes": cuda_memory_bytes,
        },
        "paper_metrics": {
            "control_success_rate": metrics.get("success_rate"),
            "num_successes": (
                int(sum(int(bool(x)) for x in metrics.get("episode_successes", [])))
                if isinstance(metrics.get("episode_successes"), list)
                else None
            ),
            "avg_return": _extract_metric(
                metrics,
                [
                    "avg_return",
                    "return",
                    "returns",
                    "episode_return",
                    "episode_returns",
                    "mean_return",
                ],
            ),
            "avg_final_distance": _extract_metric(
                metrics,
                [
                    "avg_final_distance",
                    "final_distance",
                    "distance_to_goal",
                    "goal_distance",
                    "episode_final_distance",
                ],
            ),
            "planning_time_sec": evaluation_time_sec,
            "model_parameters": count_parameters(model),
            "model_size_bytes": model_size_bytes(model),
            "compression_ratio": None,
            "interface_call_path": cost_interface,
        },
        "notes": (
            "Dataset-driven SWM evaluation using the same start/goal sampling logic "
            "as external/le-wm/eval.py. This benchmark intentionally reports only "
            "control and efficiency metrics, not representation probing or VoE metrics."
        ),
        "model_reports": _model_report_metadata(model_path),
    }

    return to_jsonable(result)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="tworoom", choices=list(ENV_SPECS.keys()))
    families = available_model_families()
    parser.add_argument(
        "--model-family",
        default="lewm_hf",
        choices=families,
        help=(
            "Model loader family from oawc.models registry. "
            f"Available: {', '.join(families)}."
        ),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()
    if args.model_path is not None:
        model_family = "torch_file"
        checkpoint = str(Path(args.model_path))
    else:
        if args.checkpoint is None:
            raise ValueError("Provide --checkpoint or --model-path.")
        model_family = args.model_family
        checkpoint = args.checkpoint

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("outputs/benchmarks") / args.env / args.tag
    )

    result = run_cost_model_benchmark(
        env_name=args.env,
        model_family=model_family,
        checkpoint=checkpoint,
        tag=args.tag,
        num_eval=args.num_eval,
        seed=args.seed,
        device=args.device,
        save_video=args.save_video,
        output_dir=output_dir,
        model_path=args.model_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.tag}_seed{args.seed}_n{args.num_eval}.json"

    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print("\nCost-model benchmark complete")
    print(f"  env:          {args.env}")
    print(f"  model_family: {model_family}")
    print(f"  checkpoint:   {checkpoint}")
    print(f"  tag:          {args.tag}")
    print(f"  num_eval:     {args.num_eval}")
    print(f"  seed:         {args.seed}")
    print(f"  device:       {result['efficiency']['cuda_memory_bytes'] is not None and 'cuda' or 'cpu'}")
    print(f"  success_rate: {result['performance']['success_rate']}")
    print(f"  eval_time_s:  {result['efficiency']['evaluation_time_sec']:.3f}")
    print(f"  parameters:   {result['efficiency']['model_parameters']}")
    print(f"  saved:        {out_path}")


if __name__ == "__main__":
    main()
