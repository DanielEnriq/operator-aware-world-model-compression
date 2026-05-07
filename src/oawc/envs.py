from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import stable_worldmodel as swm


@dataclass(frozen=True)
class EnvSpec:
    name: str
    env_id: str

    # Stable WorldModel expects image_shape=(H, W), not (C, H, W).
    image_shape: tuple[int, int] = (224, 224)

    # LeWM / SWM paper protocol.
    frame_skip: int = 5
    action_block: int = 5
    history_size: int = 3

    # Dataset-driven evaluation protocol.
    eval_budget_steps: int = 50
    goal_distance_steps: int = 25

    # CEM planning protocol.
    cem_samples: int = 300
    cem_elites: int = 30
    cem_iterations: int = 10
    cem_initial_variance: float = 1.0
    planning_horizon_blocks: int = 5
    receding_horizon_blocks: int = 5

    # Known from SWM space inspection.
    action_dim: int | None = None
    observation_kind: str | None = None

    # Optional path to LeWM repo eval config.
    lewm_eval_config: str | None = None


ENV_SPECS: dict[str, EnvSpec] = {
    "tworoom": EnvSpec(
        name="tworoom",
        env_id="swm/TwoRoom-v1",

        # Matches external/le-wm/config/eval/tworoom.yaml.
        # Important: training/data uses frameskip=5, but the eval World uses
        # frame_skip=1 and PlanConfig.action_block=5.
        history_size=1,
        frame_skip=1,
        action_block=5,

        action_dim=2,
        observation_kind="low_dim_state",
        eval_budget_steps=50,
        goal_distance_steps=25,
        cem_iterations=10,
        lewm_eval_config="external/le-wm/config/eval/tworoom.yaml",
    ),
    "pusht": EnvSpec(
        name="pusht",
        env_id="swm/PushT-v1",
        history_size=3,
        action_dim=2,
        observation_kind="dict_state_proprio",
        eval_budget_steps=50,
        goal_distance_steps=25,
        cem_iterations=30,
        lewm_eval_config="external/le-wm/config/eval/pusht.yaml",
    ),
    "ogbench_cube": EnvSpec(
        name="ogbench_cube",
        env_id="swm/OGBCube-v0",
        history_size=3,
        action_dim=5,
        observation_kind="low_dim_state",
        eval_budget_steps=50,
        goal_distance_steps=25,
        cem_iterations=10,
        lewm_eval_config="external/le-wm/config/eval/cube.yaml",
    ),
}


def get_env_spec(name: str) -> EnvSpec:
    try:
        return ENV_SPECS[name]
    except KeyError as e:
        valid = ", ".join(sorted(ENV_SPECS))
        raise KeyError(f"Unknown env '{name}'. Valid envs: {valid}") from e


def make_world(
    env_name: str,
    *,
    num_envs: int = 1,
    seed: int = 2349867,
    max_episode_steps: int | None = None,
    goal_conditioned: bool = True,
    verbose: int = 1,
    **kwargs: Any,
) -> swm.World:
    """
    Create a Stable WorldModel World using the frozen benchmark contract.

    Important:
      - image_shape is (H, W), not (C, H, W).
      - history_size and frame_skip are part of the wrapped observation contract.
      - World.reset() returns None and stores data in world.infos.
      - World.step() uses world.policy; do not treat World as raw Gym.
    """
    spec = get_env_spec(env_name)

    return swm.World(
        spec.env_id,
        num_envs=num_envs,
        image_shape=spec.image_shape,
        seed=seed,
        history_size=spec.history_size,
        frame_skip=spec.frame_skip,
        max_episode_steps=max_episode_steps or spec.eval_budget_steps,
        goal_conditioned=goal_conditioned,
        verbose=verbose,
        **kwargs,
    )


def env_specs_as_dict() -> dict[str, dict[str, Any]]:
    return {name: asdict(spec) for name, spec in ENV_SPECS.items()}


def get_planning_config_kwargs(env_name: str) -> dict[str, Any]:
    """
    Arguments for stable_worldmodel.policy.PlanConfig.

    PlanConfig signature:
      PlanConfig(horizon, receding_horizon, history_len=1, action_block=1, warm_start=True)
    """
    spec = get_env_spec(env_name)

    return {
        "horizon": spec.planning_horizon_blocks,
        "receding_horizon": spec.receding_horizon_blocks,
        "history_len": spec.history_size,
        "action_block": spec.action_block,
        "warm_start": True,
    }


def get_cem_solver_kwargs(env_name: str, *, device: str = "cpu", seed: int = 1234) -> dict[str, Any]:
    """
    Arguments for stable_worldmodel.solver.CEMSolver, excluding model.
    """
    spec = get_env_spec(env_name)

    return {
        "num_samples": spec.cem_samples,
        "var_scale": spec.cem_initial_variance,
        "n_steps": spec.cem_iterations,
        "topk": spec.cem_elites,
        "device": device,
        "seed": seed,
    }


def get_dataset_eval_kwargs(env_name: str) -> dict[str, int]:
    """
    Dataset-driven evaluation settings from the LeWM paper.

    These are used when evaluating from an offline dataset:
      goal_offset = goal_distance_steps
      eval_budget = eval_budget_steps
    """
    spec = get_env_spec(env_name)

    return {
        "goal_offset": spec.goal_distance_steps,
        "eval_budget": spec.eval_budget_steps,
    }
