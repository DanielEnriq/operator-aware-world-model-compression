from __future__ import annotations

import argparse
import os
from pathlib import Path

import stable_worldmodel as swm


EXPECTED_OBJECT_CKPTS = {
    "lewm": {
        "tworoom": "tworoom/lewm_object.ckpt",
        "pusht": "pusht/lewm_object.ckpt",
        "ogbench_cube": "cube/lewm_object.ckpt",
    },
    "pldm": {
        "tworoom": "tworoom/pldm_object.ckpt",
        "pusht": "pusht/pldm_object.ckpt",
        "ogbench_cube": "cube/pldm_object.ckpt",
    },
    "dinowm": {
        "tworoom": "tworoom/dinowm_object.ckpt",
        "pusht": "pusht/dinowm_object.ckpt",
    },
    "dinowm_noprop": {
        "tworoom": "tworoom/dinowm_noprop_object.ckpt",
        "pusht": "pusht/dinowm_noprop_object.ckpt",
        "ogbench_cube": "cube/dinowm_noprop_object.ckpt",
    },
}

HF_LEWM_REPOS = {
    "tworoom": "quentinll/lewm-tworooms",
    "pusht": "quentinll/lewm-pusht",
    "ogbench_cube": "quentinll/lewm-cube",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--try-load", action="store_true")
    args = parser.parse_args()

    cache = Path(os.environ.get("STABLEWM_HOME", swm.data.utils.get_cache_dir()))
    print(f"STABLEWM_HOME={cache}")
    print()

    print("HuggingFace LeWM repos already handled by our lewm_hf loader:")
    for env, repo in HF_LEWM_REPOS.items():
        print(f"  {env:14s} {repo}")
    print()

    print("Local SWM object checkpoint availability:")
    missing = []
    present = []

    for family, envs in EXPECTED_OBJECT_CKPTS.items():
        print(f"\n[{family}]")
        for env, rel in envs.items():
            path = cache / rel
            ok = path.exists()
            status = "OK" if ok else "MISSING"
            print(f"  {env:14s} {status:8s} {path}")
            if ok:
                present.append((family, env, rel))
            else:
                missing.append((family, env, rel))

    if args.try_load:
        print("\nTrying to load present object checkpoints with swm.policy.AutoCostModel...")
        for family, env, rel in present:
            run_name = rel.removesuffix("_object.ckpt")
            print(f"\n  {family}/{env}: {run_name}")
            try:
                model = swm.policy.AutoCostModel(run_name)
                print(f"    loaded: {type(model)}")
                print(f"    has get_cost: {hasattr(model, 'get_cost')}")
            except Exception as e:
                print(f"    LOAD FAILED: {type(e).__name__}: {e}")

    print("\nSummary:")
    print(f"  present local object checkpoints: {len(present)}")
    print(f"  missing local object checkpoints: {len(missing)}")
    if missing:
        print("\nMissing artifacts are expected until the LeWM Google Drive baseline archive is downloaded manually.")
        print("README Drive folder: https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e")


if __name__ == "__main__":
    main()
