from __future__ import annotations

import argparse
import os
from pathlib import Path

from oawc.benchmark import (
    get_dataset_name,
    load_hdf5_dataset,
    sample_dataset_eval_tasks,
)
from oawc.envs import ENV_SPECS, make_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default=None,
        choices=list(ENV_SPECS.keys()),
        help="If set, check only one benchmark environment.",
    )
    parser.add_argument("--num-eval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    print(f"STABLEWM_HOME={root}")

    env_names = [args.env] if args.env is not None else list(ENV_SPECS.keys())

    for env_name in env_names:
        spec = ENV_SPECS[env_name]
        dataset_name = get_dataset_name(env_name)

        print("\n" + "=" * 80)
        print(f"{env_name}: dataset={dataset_name}")

        dataset = load_hdf5_dataset(env_name)
        print("dataset=OK")
        print(f"  columns:       {list(dataset.column_names)}")
        print(f"  episodes:      {len(dataset.lengths)}")
        print(f"  first lengths: {dataset.lengths[:10].tolist()}")
        print(f"  num clips:     {len(dataset)}")

        eval_tasks = sample_dataset_eval_tasks(
            dataset=dataset,
            goal_offset_steps=spec.goal_distance_steps,
            num_eval=args.num_eval,
            seed=args.seed,
        )
        print("eval task sampling=OK")
        print(f"  eval_episodes: {eval_tasks['episodes_idx']}")
        print(f"  eval_starts:   {eval_tasks['start_steps']}")

        world = make_world(env_name, num_envs=1, seed=args.seed)
        world.reset(seed=args.seed)

        print("world reset=OK")
        print(f"  env_id:       {spec.env_id}")
        print(f"  info_keys:    {sorted(world.infos.keys())}")

        if "pixels" in world.infos:
            print(f"  pixels shape: {world.infos['pixels'].shape}")
        if "state" in world.infos:
            print(f"  state shape:  {world.infos['state'].shape}")
        if "action" in world.infos:
            print(f"  action shape: {world.infos['action'].shape}")

    print("\nBenchmark data check passed.")


if __name__ == "__main__":
    main()
