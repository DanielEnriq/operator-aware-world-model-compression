from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm

from oawc.benchmark import (
    get_dataset_name,
    load_hdf5_dataset,
    sample_dataset_eval_tasks,
)
from oawc.envs import ENV_SPECS, make_world


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def run_random_baseline(
    *,
    env_name: str,
    num_eval: int,
    seed: int,
    save_video: bool,
    output_dir: Path,
) -> dict[str, Any]:
    if env_name not in ENV_SPECS:
        raise KeyError(f"Unknown env={env_name}. Valid envs: {sorted(ENV_SPECS)}")

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

    # For dataset-driven evaluation, one vector env per sampled task is the cleanest contract:
    # each env receives one dataset episode/start pair.
    world = make_world(
        env_name,
        num_envs=num_eval,
        seed=seed,
        max_episode_steps=2 * spec.eval_budget_steps,
        goal_conditioned=True,
        verbose=0,
    )

    policy = swm.policy.RandomPolicy(seed=seed)
    world.set_policy(policy)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "videos"

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset_steps=spec.goal_distance_steps,
        eval_budget=spec.eval_budget_steps,
        callables=None,
        save_video=save_video,
        video_path=video_path,
    )
    end_time = time.time()

    evaluation_time_sec = end_time - start_time

    result = {
        "benchmark_version": "oawc_dataset_driven_v1",
        "model": {
            "name": "random_policy",
            "family": "random",
            "checkpoint": None,
            "compression": None,
            "is_cost_model": False,
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
            "seed": seed,
            "episodes_idx": episodes_idx,
            "start_steps": start_steps,
            "goal_offset_steps": spec.goal_distance_steps,
            "eval_budget_steps": spec.eval_budget_steps,
            "max_episode_steps": 2 * spec.eval_budget_steps,
            "save_video": save_video,
        },
        "planning": {
            "solver": None,
            "cem_samples": None,
            "cem_elites": None,
            "cem_iterations": None,
            "cem_initial_variance": None,
            "planning_horizon_blocks": None,
            "receding_horizon_blocks": None,
        },
        "performance": {
            "success_rate": metrics.get("success_rate"),
            "episode_successes": metrics.get("episode_successes"),
            "raw_metrics": metrics,
        },
        "efficiency": {
            "evaluation_time_sec": evaluation_time_sec,
            "episodes_per_sec": num_eval / evaluation_time_sec if evaluation_time_sec > 0 else None,
            "model_parameters": None,
            "model_size_bytes": None,
            "cost_model_latency_sec": None,
            "candidate_throughput_per_sec": None,
            "cuda_memory_bytes": None,
        },
        "notes": (
            "RandomPolicy baseline using the same dataset-driven SWM evaluation path "
            "that will be used for LeWM, DINO-WM, PLDM, and compressed variants. "
            "Model-specific efficiency fields are None because this baseline has no learned cost model."
        ),
    }

    return to_jsonable(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom", choices=list(ENV_SPECS.keys()))
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = Path("outputs/benchmarks") / args.env / "random_policy"
    else:
        output_dir = Path(args.output_dir)

    result = run_random_baseline(
        env_name=args.env,
        num_eval=args.num_eval,
        seed=args.seed,
        save_video=args.save_video,
        output_dir=output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"random_policy_seed{args.seed}_n{args.num_eval}.json"

    with out_path.open("w") as f:
        json.dump(result, f, indent=2)

    print("\nRandom baseline benchmark complete")
    print(f"  env:          {args.env}")
    print(f"  num_eval:     {args.num_eval}")
    print(f"  seed:         {args.seed}")
    print(f"  success_rate: {result['performance']['success_rate']}")
    print(f"  eval_time_s:  {result['efficiency']['evaluation_time_sec']:.3f}")
    print(f"  saved:        {out_path}")


if __name__ == "__main__":
    main()
