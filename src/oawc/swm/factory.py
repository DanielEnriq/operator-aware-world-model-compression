from __future__ import annotations

from oawc.config import RunConfig


def make_world(cfg: RunConfig):
    import stable_worldmodel as swm

    return swm.World(
        cfg.env_id,
        num_envs=cfg.num_envs,
        image_shape=cfg.image_shape,
        max_episode_steps=cfg.T,
        verbose=0,
        **cfg.world_kwargs,
    )


def make_policy(cfg: RunConfig):
    if cfg.policy_type == "random":
        from stable_worldmodel.policy import RandomPolicy

        return RandomPolicy(**cfg.policy_kwargs)

    if cfg.env_key == "pusht" and cfg.policy_type == "weak":
        from stable_worldmodel.envs.pusht import WeakPolicy

        return WeakPolicy(**cfg.policy_kwargs)

    if cfg.env_key == "tworoom" and cfg.policy_type == "expert":
        from stable_worldmodel.envs.two_room import ExpertPolicy

        return ExpertPolicy(**cfg.policy_kwargs)

    raise ValueError(
        f"Unsupported policy config: env_key={cfg.env_key!r}, "
        f"policy_type={cfg.policy_type!r}"
    )