from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    env_key: str
    env_id: str
    policy_type: str
    checkpoint_repo: str

    image_shape: tuple[int, int] = (224, 224)
    T: int = 100
    num_envs: int = 8
    seed: int = 42

    action_block: int = 5
    history_size: int = 3
    state_key: str = "state"

    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    world_kwargs: dict[str, Any] = field(default_factory=dict)


PHASE1_RUNS: dict[str, RunConfig] = {
    "pusht_weak": RunConfig(
        run_name="pusht_weak",
        env_key="pusht",
        env_id="swm/PushT-v1",
        policy_type="weak",
        policy_kwargs={"dist_constraint": 50},
        checkpoint_repo="quentinll/lewm-pusht",
        state_key="state",
    ),
}


def get_run_config(run_name: str) -> RunConfig:
    try:
        return PHASE1_RUNS[run_name]
    except KeyError as exc:
        valid = ", ".join(sorted(PHASE1_RUNS))
        raise ValueError(f"Unknown run_name={run_name!r}. Valid runs: {valid}") from exc