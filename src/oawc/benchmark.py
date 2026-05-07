from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

try:
    import stable_pretraining as spt
except ImportError:
    spt = None

from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.solver import CEMSolver

from oawc.envs import (
    ENV_SPECS,
    get_cem_solver_kwargs,
    get_env_spec,
    get_planning_config_kwargs,
    make_world,
)


class CostModel(Protocol):
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        ...


DATASET_NAMES: dict[str, str] = {
    # Matches external/le-wm/config/eval/tworoom.yaml
    "tworoom": "tworoom",

    # Matches external/le-wm/config/eval/pusht.yaml
    "pusht": "pusht_expert_train",

    # Matches external/le-wm/config/eval/cube.yaml
    "ogbench_cube": "ogbench/cube_single_expert",
}


DEFAULT_KEYS_TO_CACHE: dict[str, list[str]] = {
    "tworoom": [
        "pixels",
        "action",
        "state",
        "proprio",
        "goal_state",
        "distance_to_target",
        "step_idx",
        "episode_idx",
        "ep_idx",
    ],
    "pusht": [
        "pixels",
        "action",
        "state",
        "proprio",
        "goal",
        "goal_state",
        "goal_proprio",
        "block_pose",
        "pos_agent",
        "vel_agent",
        "step_idx",
        "episode_idx",
        "ep_idx",
    ],
    "ogbench_cube": [
        "pixels",
        "action",
        "observation",
        "target",
        "success",
        "proprio/effector_pos",
        "proprio/effector_yaw",
        "proprio/gripper_contact",
        "proprio/gripper_opening",
        "proprio/gripper_vel",
        "proprio/joint_pos",
        "proprio/joint_vel",
        "privileged/block_0_pos",
        "privileged/block_0_quat",
        "privileged/block_0_yaw",
        "step_idx",
        "episode_idx",
        "ep_idx",
    ],
}


def image_transform(img_size: int = 224):
    """
    Match LeWM eval preprocessing: ToImage -> float -> ImageNet norm -> Resize.

    Used by WorldModelPolicy to transform `pixels` and `goal`.
    """
    if spt is None:
        raise ImportError(
            "stable_pretraining is required for ImageNet dataset stats. "
            "Install the same SWM/LeWM environment used for evaluation."
        )

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def get_dataset_name(env_name: str) -> str:
    try:
        return DATASET_NAMES[env_name]
    except KeyError as e:
        valid = ", ".join(sorted(DATASET_NAMES))
        raise KeyError(f"Unknown env '{env_name}'. Valid envs: {valid}") from e


def load_hdf5_dataset(
    env_name: str,
    *,
    keys_to_load: list[str] | None = None,
):
    """
    Load the official SWM/LeWM HDF5 dataset for an environment.

    Important:
      Do NOT cache pixels. The official datasets are large; caching pixels can
      require >100GB RAM. We only cache small indexing columns needed for
      deterministic dataset-driven evaluation.
    """
    import os
    from pathlib import Path

    import h5py
    import stable_worldmodel as swm

    dataset_name = get_dataset_name(env_name)

    root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    h5_path = root / f"{dataset_name}.h5"

    if not h5_path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {h5_path}\n"
            f"Run scripts/fetch_lewm_datasets.py for env={env_name}."
        )

    with h5py.File(h5_path, "r") as f:
        available = set(f.keys())

    # Small columns only. Never cache pixels/action/state.
    keys_to_cache = [
        k for k in ["ep_idx", "episode_idx", "step_idx"]
        if k in available
    ]

    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
        cache_dir=root,
    )


def detect_episode_column(dataset: Any) -> str:
    if "episode_idx" in dataset.column_names:
        return "episode_idx"
    if "ep_idx" in dataset.column_names:
        return "ep_idx"
    raise KeyError(
        "Dataset has neither 'episode_idx' nor 'ep_idx'. "
        f"Available columns: {dataset.column_names}"
    )


def get_episode_lengths(dataset: Any, episode_ids: np.ndarray) -> np.ndarray:
    """
    Match external/le-wm/eval.py: compute episode lengths from step_idx.
    """
    ep_col = detect_episode_column(dataset)
    episode_idx = dataset.get_col_data(ep_col)
    step_idx = dataset.get_col_data("step_idx")

    lengths = []
    for ep_id in episode_ids:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)

    return np.asarray(lengths)


def sample_dataset_eval_tasks(
    dataset: Any,
    *,
    goal_offset_steps: int,
    num_eval: int,
    seed: int,
) -> dict[str, list[int]]:
    """
    Sample dataset-driven evaluation tasks the same way as external/le-wm/eval.py.

    A task is:
      episode_idx = e
      start_step = s
      goal_step = s + goal_offset_steps

    We only sample starts where that future goal exists.
    """
    ep_col = detect_episode_column(dataset)

    ep_indices, _ = np.unique(dataset.get_col_data(ep_col), return_index=True)
    episode_len = get_episode_lengths(dataset, ep_indices)

    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i]
        for i, ep_id in enumerate(ep_indices)
    }

    episode_per_row = dataset.get_col_data(ep_col)
    step_per_row = dataset.get_col_data("step_idx")

    max_start_per_row = np.asarray(
        [max_start_idx_dict[ep_id] for ep_id in episode_per_row]
    )

    valid_mask = step_per_row <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    if len(valid_indices) < num_eval:
        raise ValueError(
            f"Not enough valid starts for eval: requested {num_eval}, "
            f"found {len(valid_indices)}."
        )

    rng = np.random.default_rng(seed)

    chosen_positions = rng.choice(
        len(valid_indices) - 1,
        size=num_eval,
        replace=False,
    )

    row_indices = np.sort(valid_indices[chosen_positions])
    rows = dataset.get_row_data(row_indices)

    return {
        "row_indices": row_indices.tolist(),
        "episodes_idx": rows[ep_col].astype(int).tolist(),
        "start_steps": rows["step_idx"].astype(int).tolist(),
        "goal_offset_steps": int(goal_offset_steps),
    }


def fit_world_policy_processors(
    dataset: Any,
    *,
    keys_to_process: list[str],
) -> dict[str, preprocessing.StandardScaler]:
    """
    Fit StandardScaler objects for non-pixel columns, matching external/le-wm/eval.py.

    These processors are passed to WorldModelPolicy(process=...).
    """
    process = {}

    for col in keys_to_process:
        if col == "pixels":
            continue
        if col not in dataset.column_names:
            continue

        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)

        if col_data.ndim == 1:
            col_data = col_data[:, None]

        col_data = col_data[~np.isnan(col_data).any(axis=1)]

        if len(col_data) == 0:
            continue

        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    return process


def make_planning_policy(
    env_name: str,
    cost_model: CostModel,
    *,
    dataset: Any | None = None,
    device: str = "cpu",
    solver_seed: int = 1234,
    process_keys: list[str] | None = None,
) -> WorldModelPolicy:
    """
    Construct the official SWM model-based planning policy.

    The cost model must implement:
      get_cost(info_dict, action_candidates) -> costs
    """
    solver = CEMSolver(
        model=cost_model,
        **get_cem_solver_kwargs(env_name, device=device, seed=solver_seed),
    )

    plan_config = PlanConfig(**get_planning_config_kwargs(env_name))

    transform = {
        "pixels": image_transform(224),
        "goal": image_transform(224),
    }

    process = {}
    if dataset is not None:
        process_keys = process_keys or DEFAULT_KEYS_TO_CACHE[env_name]
        process = fit_world_policy_processors(dataset, keys_to_process=process_keys)

    return WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )


def evaluate_from_dataset(
    env_name: str,
    cost_model: CostModel,
    *,
    num_eval: int,
    seed: int = 0,
    device: str = "cpu",
    num_envs: int = 1,
    dataset_name: str | None = None,
    cache_dir: str | Path | None = None,
    save_video: bool = True,
    video_path: str | Path = "outputs/videos",
    callables: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Official dataset-driven benchmark evaluation.

    This matches the LeWM paper protocol:
      - sample start states from an offline dataset
      - set goal as a future state from the same trajectory
      - run MPC/CEM under a fixed eval budget
      - report success rate and metrics from SWM evaluate_from_dataset
    """
    spec = get_env_spec(env_name)

    dataset = load_hdf5_dataset(
        env_name,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )

    tasks = sample_dataset_eval_tasks(
        dataset,
        goal_offset_steps=spec.goal_distance_steps,
        num_eval=num_eval,
        seed=seed,
    )

    world = make_world(
        env_name,
        num_envs=num_envs,
        max_episode_steps=2 * spec.eval_budget_steps,
        verbose=0,
    )

    policy = make_planning_policy(
        env_name,
        cost_model,
        dataset=dataset,
        device=device,
        solver_seed=seed,
    )

    world.set_policy(policy)

    video_path = Path(video_path) / env_name
    video_path.mkdir(parents=True, exist_ok=True)

    start = time.time()

    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=tasks["start_steps"],
        goal_offset_steps=spec.goal_distance_steps,
        eval_budget=spec.eval_budget_steps,
        episodes_idx=tasks["episodes_idx"],
        callables=callables,
        save_video=save_video,
        video_path=video_path,
    )

    elapsed = time.time() - start

    result = {
        "env_name": env_name,
        "dataset_name": dataset_name or get_dataset_name(env_name),
        "mode": "dataset_driven",
        "num_eval": int(num_eval),
        "seed": int(seed),
        "device": device,
        "env_spec": asdict(spec),
        "tasks": tasks,
        "metrics": metrics,
        "elapsed_sec": float(elapsed),
        "episodes_per_sec": float(num_eval / elapsed) if elapsed > 0 else None,
    }

    return result


def measure_cost_model_latency(
    env_name: str,
    cost_model: CostModel,
    *,
    seed: int = 0,
    device: str = "cpu",
    num_envs: int = 1,
    num_candidates: int | None = None,
    horizon: int | None = None,
    repeats: int = 10,
) -> dict[str, Any]:
    """
    Systems benchmark for cost-model planning latency.

    Uses a real SWM info_dict after reset and random candidate action sequences.
    This does not measure success rate; it measures the operator that CEM calls.
    """
    spec = get_env_spec(env_name)

    num_candidates = num_candidates or spec.cem_samples
    horizon = horizon or spec.planning_horizon_blocks

    world = make_world(env_name, num_envs=num_envs, verbose=0)
    world.reset(seed=seed)

    action_dim = spec.action_dim
    if action_dim is None:
        action_dim = int(world.envs.action_space.shape[-1])

    rng = np.random.default_rng(seed)
    action_candidates = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(num_envs, num_candidates, horizon, action_dim),
    ).astype(np.float32)

    action_candidates_t = torch.tensor(
        action_candidates,
        dtype=torch.float32,
        device=device,
    )

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = cost_model.get_cost(world.infos, action_candidates_t)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    costs = None

    for _ in range(repeats):
        start = time.time()

        with torch.no_grad():
            costs = cost_model.get_cost(world.infos, action_candidates_t)

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        times.append(time.time() - start)

    mean_latency = float(np.mean(times))
    total_candidates = num_envs * num_candidates

    return {
        "env_name": env_name,
        "device": device,
        "num_envs": int(num_envs),
        "num_candidates": int(num_candidates),
        "horizon": int(horizon),
        "action_dim": int(action_dim),
        "repeats": int(repeats),
        "costs_shape": list(costs.shape) if costs is not None else None,
        "latency_mean_sec": mean_latency,
        "latency_std_sec": float(np.std(times)),
        "candidate_costs_per_sec": float(total_candidates / mean_latency),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else None
        ),
    }


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def model_size_bytes(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return int(total)


def save_benchmark_result(path: str | Path, result: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    path.write_text(json.dumps(result, indent=2, default=default))


def export_benchmark_contract(path: str | Path = "outputs/benchmark_contract.json") -> None:
    contract = {
        "datasets": DATASET_NAMES,
        "env_specs": {name: asdict(spec) for name, spec in ENV_SPECS.items()},
        "keys_to_cache": DEFAULT_KEYS_TO_CACHE,
        "model_contract": "cost_model.get_cost(info_dict, action_candidates)",
        "world_eval_method": "World.evaluate_from_dataset",
        "dataset_class": "stable_worldmodel.data.HDF5Dataset",
    }
    save_benchmark_result(path, contract)
