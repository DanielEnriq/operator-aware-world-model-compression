from __future__ import annotations

import argparse
import time
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

from oawc.config import get_run_config
from oawc.paths import phase1_dirs, save_json
from oawc.swm.factory import make_policy, make_world


def collect_rollout(run_name: str, overwrite: bool = False) -> None:
    cfg = get_run_config(run_name)
    dirs = phase1_dirs(cfg)

    obs_path = dirs["data"] / "obs.npy"
    states_path = dirs["data"] / "states.npy"
    actions_path = dirs["data"] / "actions.npy"
    terminateds_path = dirs["data"] / "terminateds.npy"
    truncateds_path = dirs["data"] / "truncateds.npy"
    metadata_path = dirs["data"] / "metadata.json"

    if obs_path.exists() and states_path.exists() and actions_path.exists() and not overwrite:
        print(f"Rollout already exists for {run_name}. Use --overwrite to recollect.")
        print(f"Data dir: {dirs['data']}")
        return

    world = make_world(cfg)
    policy = make_policy(cfg)
    world.set_policy(policy)

    world.reset(seed=cfg.seed)

    obs_list = []
    state_list = []
    action_list = []
    terminated_list = []
    truncated_list = []

    for _ in tqdm(range(cfg.T), desc=f"Collecting {cfg.run_name}"):
        world.step()

        obs_list.append(world.infos["pixels"][:, 0].copy())
        state_list.append(world.infos[cfg.state_key][:, 0].copy())
        action_list.append(world.infos["action"][:, 0].copy())
        terminated_list.append(world.infos["terminated"][:, 0].copy())
        truncated_list.append(world.infos["truncated"][:, 0].copy())

    world.close()

    obs = np.asarray(obs_list)
    states = np.asarray(state_list)
    actions = np.asarray(action_list)
    terminateds = np.asarray(terminated_list)
    truncateds = np.asarray(truncated_list)

    np.save(obs_path, obs)
    np.save(states_path, states)
    np.save(actions_path, actions)
    np.save(terminateds_path, terminateds)
    np.save(truncateds_path, truncateds)

    metadata = {
        "config": asdict(cfg),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "obs_shape": list(obs.shape),
        "states_shape": list(states.shape),
        "actions_shape": list(actions.shape),
        "terminateds_shape": list(terminateds.shape),
        "truncateds_shape": list(truncateds.shape),
        "obs_dtype": str(obs.dtype),
        "states_dtype": str(states.dtype),
        "actions_dtype": str(actions.dtype),
    }

    save_json(metadata_path, metadata)

    print("Saved rollout artifacts:")
    print(f"  obs:         {obs_path} {obs.shape} {obs.dtype}")
    print(f"  states:      {states_path} {states.shape} {states.dtype}")
    print(f"  actions:     {actions_path} {actions.shape} {actions.dtype}")
    print(f"  metadata:    {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    collect_rollout(args.run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()