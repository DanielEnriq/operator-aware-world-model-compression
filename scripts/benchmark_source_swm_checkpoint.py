from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm
import torch

from oawc.benchmark import (
    count_parameters,
    fit_world_policy_processors,
    get_dataset_name,
    image_transform,
    load_hdf5_dataset,
    model_size_bytes,
    sample_dataset_eval_tasks,
)
from oawc.envs import (
    ENV_SPECS,
    get_cem_solver_kwargs,
    get_eval_callables,
    get_planning_config_kwargs,
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
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def _ensure_source_swm_imported(project_root: Path) -> Path:
    swm_file = Path(swm.__file__).resolve()
    expected_prefix = (
        project_root / "external" / "stable-worldmodel"
    ).resolve()
    print(f"stable_worldmodel.__file__: {swm_file}")
    if expected_prefix not in swm_file.parents:
        raise RuntimeError(
            "Expected source SWM import from external/stable-worldmodel, got: "
            f"{swm_file}"
        )
    return swm_file


def _resolve_object_checkpoint(
    checkpoint: str,
    stablewm_home: Path,
) -> Path:
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
        generic_ckpts = sorted(run_dir.glob("*.ckpt"))
        if generic_ckpts:
            return generic_ckpts[0]

    raise FileNotFoundError(
        "Unable to resolve source-SWM object checkpoint for "
        f"'{checkpoint}' under {stablewm_home / 'checkpoints'}"
    )


def _load_cost_model_from_object_checkpoint(
    object_ckpt: Path,
) -> torch.nn.Module:
    loaded = torch.load(object_ckpt, map_location="cpu", weights_only=False)
    model = getattr(loaded, "model", loaded)
    if not hasattr(model, "get_cost"):
        raise TypeError(
            f"Loaded object from {object_ckpt} does not expose get_cost()."
        )
    model = model.eval()
    model.requires_grad_(False)
    return model


def _make_source_world(
    *,
    env_name: str,
    num_envs: int,
    max_episode_steps: int,
) -> Any:
    spec = ENV_SPECS[env_name]
    return swm.World(
        spec.env_id,
        num_envs=num_envs,
        image_shape=spec.image_shape,
        max_episode_steps=max_episode_steps,
        goal_conditioned=True,
    )


def _resolve_planner_settings(args: argparse.Namespace) -> dict[str, int]:
    if args.smoke_planner:
        return {
            "cem_samples": 8,
            "cem_elites": 2,
            "cem_iterations": 1,
            "planning_horizon_blocks": 2,
            "receding_horizon_blocks": 1,
        }
    return {
        "cem_samples": args.cem_samples,
        "cem_elites": args.cem_elites,
        "cem_iterations": args.cem_iterations,
        "planning_horizon_blocks": args.planning_horizon_blocks,
        "receding_horizon_blocks": args.receding_horizon_blocks,
    }


def _make_source_planning_policy(
    *,
    env_name: str,
    cost_model: torch.nn.Module,
    dataset: Any,
    device: str,
    seed: int,
    planner: dict[str, int],
) -> Any:
    process = fit_world_policy_processors(
        dataset,
        keys_to_process=["action", "proprio"],
    )
    transform = {
        "pixels": image_transform(224),
        "goal": image_transform(224),
    }

    cem_kwargs = get_cem_solver_kwargs(env_name, device=device, seed=seed)
    cem_kwargs["num_samples"] = planner["cem_samples"]
    cem_kwargs["topk"] = planner["cem_elites"]
    cem_kwargs["n_steps"] = planner["cem_iterations"]
    solver = swm.solver.CEMSolver(model=cost_model, **cem_kwargs)

    plan_kwargs = get_planning_config_kwargs(env_name)
    plan_kwargs["horizon"] = planner["planning_horizon_blocks"]
    plan_kwargs["receding_horizon"] = planner["receding_horizon_blocks"]
    plan_config = swm.policy.PlanConfig(**plan_kwargs)

    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )


def _evaluate_with_source_api(
    *,
    world: Any,
    dataset: Any,
    episodes_idx: list[int],
    start_steps: list[int],
    goal_offset_steps: int,
    eval_budget_steps: int,
    callables: list[dict[str, Any]] | None,
    save_video: bool,
    video_path: Path,
) -> tuple[dict[str, Any], str]:
    if hasattr(world, "evaluate_from_dataset"):
        metrics = world.evaluate_from_dataset(
            dataset,
            episodes_idx=episodes_idx,
            start_steps=start_steps,
            goal_offset_steps=goal_offset_steps,
            eval_budget=eval_budget_steps,
            callables=callables,
            save_video=save_video,
            video_path=video_path,
        )
        return metrics, "evaluate_from_dataset"

    if hasattr(world, "evaluate"):
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=episodes_idx,
            start_steps=start_steps,
            goal_offset=goal_offset_steps,
            eval_budget=eval_budget_steps,
            callables=callables,
            video=video_path if save_video else None,
        )
        return metrics, "evaluate(dataset=...)"

    if hasattr(world, "_evaluate_from_dataset"):
        metrics = world._evaluate_from_dataset(  # noqa: SLF001
            dataset,
            episodes_idx,
            start_steps,
            goal_offset_steps,
            eval_budget_steps,
            callables,
            video_path if save_video else None,
            "wait",
        )
        return metrics, "_evaluate_from_dataset"

    raise AttributeError(
        "No dataset evaluation API available on source SWM World instance."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=list(ENV_SPECS.keys()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--num-eval", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cem-samples", type=int, default=300)
    parser.add_argument("--cem-elites", type=int, default=30)
    parser.add_argument("--cem-iterations", type=int, default=10)
    parser.add_argument("--planning-horizon-blocks", type=int, default=5)
    parser.add_argument("--receding-horizon-blocks", type=int, default=5)
    parser.add_argument("--smoke-planner", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    source_swm_file = _ensure_source_swm_imported(project_root)

    stablewm_home = Path(
        os.environ.get("STABLEWM_HOME", str(project_root / ".swm_cache"))
    ).expanduser()
    stablewm_home.mkdir(parents=True, exist_ok=True)

    dataset_name = get_dataset_name(args.env)
    object_ckpt = _resolve_object_checkpoint(args.checkpoint, stablewm_home)

    if args.device == "cpu" and not args.smoke_planner:
        raise ValueError(
            "CPU runs must use --smoke-planner to avoid heavy full CEM usage."
        )

    planner = _resolve_planner_settings(args)
    print("[progress] model load: start")
    cost_model = _load_cost_model_from_object_checkpoint(object_ckpt)
    print("[progress] model load: finish")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    cost_model = cost_model.to(device)

    print("[progress] dataset load: start")
    dataset = load_hdf5_dataset(args.env)
    print("[progress] dataset load: finish")
    print("[progress] task sampling: start")
    tasks = sample_dataset_eval_tasks(
        dataset=dataset,
        goal_offset_steps=ENV_SPECS[args.env].goal_distance_steps,
        num_eval=args.num_eval,
        seed=args.seed,
    )
    print("[progress] task sampling: finish")

    print("[progress] policy creation: start")
    print(
        "  CEM settings: "
        f"samples={planner['cem_samples']}, "
        f"elites={planner['cem_elites']}, "
        f"iterations={planner['cem_iterations']}, "
        f"horizon_blocks={planner['planning_horizon_blocks']}, "
        f"receding_horizon_blocks={planner['receding_horizon_blocks']}"
    )
    policy = _make_source_planning_policy(
        env_name=args.env,
        cost_model=cost_model,
        dataset=dataset,
        device=device,
        seed=args.seed,
        planner=planner,
    )
    print("[progress] policy creation: finish")

    print("[progress] world creation: start")
    world = _make_source_world(
        env_name=args.env,
        num_envs=args.num_eval,
        max_episode_steps=2 * ENV_SPECS[args.env].eval_budget_steps,
    )
    world.set_policy(policy)
    print("[progress] world creation: finish")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/benchmarks_source_swm") / args.env / args.tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "videos"

    print("[progress] evaluation start")
    start = time.time()
    metrics, eval_api_used = _evaluate_with_source_api(
        world=world,
        dataset=dataset,
        episodes_idx=tasks["episodes_idx"],
        start_steps=tasks["start_steps"],
        goal_offset_steps=ENV_SPECS[args.env].goal_distance_steps,
        eval_budget_steps=ENV_SPECS[args.env].eval_budget_steps,
        callables=get_eval_callables(args.env),
        save_video=args.save_video,
        video_path=video_path,
    )
    elapsed = time.time() - start
    print("[progress] evaluation finish")

    result = _to_jsonable(
        {
            "env": args.env,
            "checkpoint": args.checkpoint,
            "resolved_checkpoint_path": str(object_ckpt),
            "dataset_name": dataset_name,
            "num_eval": args.num_eval,
            "seed": args.seed,
            "device": device,
            "evaluation_api_used": eval_api_used,
            "source_swm_path": str(source_swm_file),
            "has_get_cost": hasattr(cost_model, "get_cost"),
            "success_rate": metrics.get("success_rate"),
            "episode_successes": metrics.get("episode_successes"),
            "raw_metrics": metrics,
            "evaluation_time_sec": elapsed,
            "parameters": count_parameters(cost_model),
            "model_size_bytes": model_size_bytes(cost_model),
        }
    )

    out_path = output_dir / f"{args.tag}_seed{args.seed}_n{args.num_eval}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\nSource-SWM checkpoint benchmark complete")
    print(f"  env:                 {args.env}")
    print(f"  checkpoint:          {args.checkpoint}")
    print(f"  resolved_path:       {object_ckpt}")
    print(f"  source_swm_path:     {source_swm_file}")
    print(f"  evaluation_api_used: {eval_api_used}")
    print(f"  success_rate:        {result['success_rate']}")
    print(f"  eval_time_s:         {elapsed:.3f}")
    print(f"  parameters:          {result['parameters']}")
    print(f"  model_size_bytes:    {result['model_size_bytes']}")
    print(f"  saved:               {out_path}")


if __name__ == "__main__":
    main()
