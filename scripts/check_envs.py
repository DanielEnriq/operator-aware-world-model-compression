from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from oawc.envs import ENV_SPECS, env_specs_as_dict, make_world


def safe_shape(space) -> str | None:
    return str(getattr(space, "shape", None))


def summarize_info_dict(info: dict) -> dict:
    summary = {}

    for key, value in info.items():
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)

        summary[key] = {
            "type": type(value).__name__,
            "shape": None if shape is None else list(shape),
            "dtype": None if dtype is None else str(dtype),
        }

    return summary


def check_env(name: str) -> dict:
    spec = ENV_SPECS[name]

    result = {
        "spec": env_specs_as_dict()[name],
        "instantiate_ok": False,
        "spaces_ok": False,
        "reset_ok": False,
        "instantiate_error": None,
        "reset_error": None,
        "action_space": None,
        "action_shape": None,
        "observation_space": None,
        "observation_shape": None,
        "info_keys_after_reset": None,
        "info_summary_after_reset": None,
    }

    try:
        world = make_world(name, num_envs=1, verbose=1)
        result["instantiate_ok"] = True

        action_space = getattr(world.envs, "action_space", None)
        observation_space = getattr(world.envs, "observation_space", None)

        result["action_space"] = str(action_space)
        result["action_shape"] = safe_shape(action_space)
        result["observation_space"] = str(observation_space)
        result["observation_shape"] = safe_shape(observation_space)
        result["spaces_ok"] = action_space is not None and observation_space is not None

        try:
            world.reset(seed=0)
            result["reset_ok"] = True
            result["info_keys_after_reset"] = sorted(list(world.infos.keys()))
            result["info_summary_after_reset"] = summarize_info_dict(world.infos)
        except Exception as e:
            result["reset_error"] = repr(e)

    except Exception as e:
        result["instantiate_error"] = repr(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/benchmark_env_check.json")
    args = parser.parse_args()

    print(f"STABLEWM_HOME={os.environ.get('STABLEWM_HOME')}")

    results = {name: check_env(name) for name in ENV_SPECS}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    for name, r in results.items():
        inst = "OK" if r["instantiate_ok"] else "FAIL"
        reset = "OK" if r["reset_ok"] else "FAIL"

        print(f"{name:<14} instantiate={inst:<4} reset={reset:<4} {r['spec']['env_id']}")
        print(f"  action_space:      {r['action_space']}")
        print(f"  observation_space: {r['observation_space']}")

        if r["info_keys_after_reset"]:
            print(f"  info_keys:         {r['info_keys_after_reset']}")

        if r["reset_error"]:
            print(f"  reset_error:       {r['reset_error']}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
