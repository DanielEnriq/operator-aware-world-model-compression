from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    ("stable_worldmodel.wm.prejepa", "PreJEPA"),
    ("stable_worldmodel.wm.prejepa.module", "CausalPredictor"),
    ("stable_worldmodel.wm.prejepa.module", "Embedder"),
    ("stable_worldmodel.wm.pldm", "PLDM"),
    ("stable_worldmodel.wm.pldm.module", "MLP"),
    ("stable_worldmodel.wm.pldm.module", "Embedder"),
    ("stable_worldmodel.wm.pldm.module", "Predictor"),
    ("stable_worldmodel.wm.loss", "PLDMLoss"),
    ("stable_worldmodel.wm.loss", "TemporalStraighteningLoss"),
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stablewm_home() -> Path:
    return Path(
        os.environ.get("STABLEWM_HOME", _project_root() / ".swm_cache")
    ).expanduser()


def _check_path(path: Path, label: str, failures: list[str]) -> None:
    if path.exists():
        print(f"[OK] {label}: {path}")
    else:
        msg = f"[MISSING] {label}: {path}"
        print(msg)
        failures.append(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-installed",
        action="store_true",
        help=(
            "Allow stable_worldmodel imports from site-packages. "
            "By default this script requires external/stable-worldmodel."
        ),
    )
    args = parser.parse_args()

    failures: list[str] = []
    project_root = _project_root()
    stablewm_home = _stablewm_home()
    stablewm_src = project_root / "external" / "stable-worldmodel"

    print("=== SWM Training Setup Check ===")
    print(f"python executable: {sys.executable}")
    print(f"project root:      {project_root}")
    print(f"STABLEWM_HOME:     {stablewm_home}")
    print(f"PYTHONPATH:        {os.environ.get('PYTHONPATH', '<unset>')}")
    print()

    train_dir = (
        project_root / "external" / "stable-worldmodel" / "scripts" / "train"
    )
    config_dir = train_dir / "config"
    data_dir = config_dir / "data"

    _check_path(
        train_dir / "prejepa.py",
        "upstream train script (prejepa)",
        failures,
    )
    _check_path(
        train_dir / "pldm.py",
        "upstream train script (pldm)",
        failures,
    )
    _check_path(
        config_dir / "prejepa.yaml",
        "upstream train config (prejepa)",
        failures,
    )
    _check_path(
        config_dir / "pldm.yaml",
        "upstream train config (pldm)",
        failures,
    )
    _check_path(
        data_dir / "tworoom.yaml",
        "upstream data config (tworoom)",
        failures,
    )
    _check_path(
        data_dir / "pusht.yaml",
        "upstream data config (pusht)",
        failures,
    )
    _check_path(
        data_dir / "ogb.yaml",
        "upstream data config (ogb)",
        failures,
    )
    print()

    try:
        swm = importlib.import_module("stable_worldmodel")
    except Exception as exc:
        failures.append(f"Failed to import stable_worldmodel: {exc}")
        swm = None
        print(f"[FAIL] import stable_worldmodel: {type(exc).__name__}: {exc}")
    else:
        swm_file = Path(getattr(swm, "__file__", ""))
        print(f"stable_worldmodel.__file__: {swm_file}")
        if not args.allow_installed:
            try:
                rel = swm_file.resolve().is_relative_to(stablewm_src.resolve())
            except Exception:
                rel = False
            if not rel:
                msg = (
                    "stable_worldmodel is not imported from external/stable-worldmodel. "
                    "Set PYTHONPATH to prioritize external source."
                )
                failures.append(msg)
                print(f"[FAIL] {msg}")
            else:
                print(
                    "[OK] stable_worldmodel import source is "
                    "external/stable-worldmodel"
                )
        print()

    for module_name, symbol_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            getattr(module, symbol_name)
        except Exception as exc:
            msg = (
                f"[FAIL] import {module_name}.{symbol_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            print(msg)
            failures.append(msg)
        else:
            print(f"[OK] import {module_name}.{symbol_name}")

    print()
    _check_path(stablewm_home / "tworoom.h5", "dataset (tworoom)", failures)
    _check_path(stablewm_home / "pusht_expert_train.h5", "dataset (pusht)", failures)
    _check_path(
        stablewm_home / "ogbench" / "cube_single_expert.h5",
        "dataset (ogbench_cube)",
        failures,
    )

    if failures:
        print("\n=== RESULT: FAILED ===")
        print("Actionable next steps:")
        print(
            "1) Clone SWM source:\n"
            "   git clone https://github.com/galilai-group/stable-worldmodel "
            "external/stable-worldmodel"
        )
        print(
            "2) Set source-first environment:\n"
            "   export PYTHONPATH=\"$PWD/external/stable-worldmodel:$PWD/src\"\n"
            "   export STABLEWM_HOME=\"$PWD/.swm_cache\""
        )
        print(
            "3) Fetch datasets:\n"
            "   uv run python scripts/fetch_lewm_datasets.py --datasets "
            "tworoom pusht ogbench_cube"
        )
        raise SystemExit(1)

    print("\n=== RESULT: PASSED ===")


if __name__ == "__main__":
    main()
