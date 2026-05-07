from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ENV_DATASET_FILES = {
    "tworoom": "tworoom.h5",
    "pusht": "pusht_expert_train.h5",
    "ogbench_cube": "ogbench/cube_single_expert.h5",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_train_config(family: str, env_name: str) -> dict[str, Any]:
    cfg_path = (
        project_root() / "configs" / "train" / family / f"{env_name}.yaml"
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing train config: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def default_stablewm_home(cfg: dict[str, Any], root: Path) -> Path:
    default_rel = cfg.get("default_stablewm_home", ".swm_cache")
    if Path(default_rel).is_absolute():
        return Path(default_rel)
    return root / default_rel


def output_model_name(family: str, env_name: str, seed: int, mode: str) -> str:
    base = f"oawc_{family}_{env_name}_seed{seed}"
    if mode == "smoke":
        return f"{base}_smoke"
    return base


def quote_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def resolve_training_settings(
    cfg: dict[str, Any],
    mode: str,
    device: str,
    max_epochs: int | None,
    batch_size: int | None,
    limit_train_batches: int | None,
    limit_val_batches: int | None,
) -> dict[str, Any]:
    settings = dict(cfg["training"][mode])

    if max_epochs is not None:
        settings["max_epochs"] = max_epochs
    if batch_size is not None:
        settings["batch_size"] = batch_size
    if limit_train_batches is not None:
        settings["limit_train_batches"] = limit_train_batches
    if limit_val_batches is not None:
        settings["limit_val_batches"] = limit_val_batches

    if device == "cpu":
        settings["accelerator"] = "cpu"
        settings["devices"] = 1
        settings["precision"] = "32"
    else:
        settings["accelerator"] = "gpu"
        settings.setdefault("devices", "auto")

    return settings


def ensure_required_paths(root: Path, stablewm_home: Path, env_name: str) -> None:
    swm_src = root / "external" / "stable-worldmodel"
    if not swm_src.exists():
        raise FileNotFoundError(
            f"Missing upstream source tree: {swm_src}\n"
            "Clone stable-worldmodel into external/stable-worldmodel."
        )

    dataset_path = stablewm_home / ENV_DATASET_FILES[env_name]
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing dataset for env={env_name}: {dataset_path}\n"
            "Run scripts/fetch_lewm_datasets.py first."
        )


def ensure_upstream_dataset_layout(
    *,
    stablewm_home: Path,
    env_name: str,
) -> Path:
    """
    Ensure datasets exist at the location expected by swm.data.load_dataset:
    <STABLEWM_HOME>/datasets/<dataset_relpath>.
    """
    dataset_rel = Path(ENV_DATASET_FILES[env_name])
    source_path = stablewm_home / dataset_rel
    datasets_root = stablewm_home / "datasets"
    target_path = datasets_root / dataset_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        return target_path

    try:
        target_path.symlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, target_path)

    return target_path


def build_hydra_command(
    cfg: dict[str, Any],
    *,
    stablewm_home: Path,
    run_id: str,
    out_name: str,
    seed: int,
    settings: dict[str, Any],
) -> list[str]:
    upstream_script = str(project_root() / cfg["upstream_script"])
    command = [sys.executable, upstream_script]

    overrides: list[str] = [
        f"output_model_name={out_name}",
        f"subdir={run_id}",
        f"seed={seed}",
        f"cache_dir={stablewm_home}",
        "wandb.enabled=false",
    ]

    if cfg["model_family"] == "swm_prejepa":
        overrides.extend(
            [
                f"dataset_name={cfg['dataset_name']}",
                f"batch_size={settings['batch_size']}",
                f"num_workers={settings['num_workers']}",
            ]
        )
    elif cfg["model_family"] == "swm_pldm":
        overrides.extend(
            [
                f"data={cfg['upstream_data_config']}",
                f"loader.batch_size={settings['batch_size']}",
                f"num_workers={settings['num_workers']}",
            ]
        )
    else:
        raise ValueError(f"Unsupported model family: {cfg['model_family']}")

    overrides.extend(
        [
            f"trainer.max_epochs={settings['max_epochs']}",
            f"trainer.accelerator={settings['accelerator']}",
            f"trainer.devices={settings['devices']}",
            f"trainer.precision={settings['precision']}",
        ]
    )

    if settings.get("limit_train_batches") is not None:
        overrides.append(
            f"+trainer.limit_train_batches={settings['limit_train_batches']}"
        )
    if settings.get("limit_val_batches") is not None:
        overrides.append(
            f"+trainer.limit_val_batches={settings['limit_val_batches']}"
        )

    command.extend(overrides)
    return command


def collect_candidate_artifacts(
    stablewm_home: Path,
    out_name: str,
    run_id: str,
) -> dict[str, Any]:
    checkpoints_root = stablewm_home / "checkpoints"
    pretrained_dir = checkpoints_root / out_name
    object_dir = checkpoints_root / run_id

    pretrained_files = (
        sorted(str(p) for p in pretrained_dir.glob("*"))
        if pretrained_dir.exists()
        else []
    )
    object_files = (
        sorted(str(p) for p in object_dir.glob("*"))
        if object_dir.exists()
        else []
    )

    object_ckpts = [p for p in object_files if p.endswith(".ckpt")]
    pt_weights = [p for p in pretrained_files if p.endswith(".pt")]

    return {
        "pretrained_dir": str(pretrained_dir),
        "pretrained_files": pretrained_files,
        "object_dir": str(object_dir),
        "object_files": object_files,
        "object_ckpts": object_ckpts,
        "weights_pt": pt_weights,
    }


def copy_selected_artifacts(src_info: dict[str, Any], target_dir: Path) -> list[str]:
    copied: list[str] = []
    target_dir.mkdir(parents=True, exist_ok=True)

    for key in ("pretrained_files", "object_files"):
        for src in src_info.get(key, []):
            src_path = Path(src)
            if src_path.is_file():
                dst = target_dir / src_path.name
                shutil.copy2(src_path, dst)
                copied.append(str(dst))
    return copied


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=["swm_prejepa", "swm_pldm"],
        required=True,
    )
    parser.add_argument(
        "--env",
        choices=["tworoom", "pusht", "ogbench_cube"],
        required=True,
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stablewm-home", default=None)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--drive-output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--copy-artifacts", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    root = project_root()
    cfg = load_train_config(args.family, args.env)

    stablewm_home = (
        Path(args.stablewm_home).expanduser()
        if args.stablewm_home
        else default_stablewm_home(cfg, root).expanduser()
    )
    stablewm_home.mkdir(parents=True, exist_ok=True)

    ensure_required_paths(root, stablewm_home, args.env)
    ensure_upstream_dataset_layout(stablewm_home=stablewm_home, env_name=args.env)

    out_name = output_model_name(args.family, args.env, args.seed, args.mode)
    run_id = args.run_id or out_name

    settings = resolve_training_settings(
        cfg=cfg,
        mode=args.mode,
        device=args.device,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    run_root = output_root / args.family / args.env / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    cmd = build_hydra_command(
        cfg,
        stablewm_home=stablewm_home,
        run_id=run_id,
        out_name=out_name,
        seed=args.seed,
        settings=settings,
    )

    swm_train_dir = (
        root / "external" / "stable-worldmodel" / "scripts" / "train"
    )
    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(stablewm_home)
    source_pythonpath = f"{root / 'external' / 'stable-worldmodel'}:{root / 'src'}"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{source_pythonpath}:{existing_pythonpath}"
        if existing_pythonpath
        else source_pythonpath
    )

    print("Resolved training command:")
    print(quote_cmd(cmd))
    print(f"Working directory: {swm_train_dir}")
    print(f"STABLEWM_HOME: {env['STABLEWM_HOME']}")
    print(f"PYTHONPATH: {env['PYTHONPATH']}")

    launched = False
    return_code = None
    if not args.dry_run:
        launched = True
        proc = subprocess.run(cmd, cwd=swm_train_dir, env=env, check=False)
        return_code = proc.returncode
        if return_code != 0:
            print(f"Training process exited with code {return_code}.")

    artifacts = collect_candidate_artifacts(
        stablewm_home=stablewm_home,
        out_name=out_name,
        run_id=run_id,
    )
    copied_local: list[str] = []
    copied_drive: list[str] = []

    if args.copy_artifacts:
        copied_local = copy_selected_artifacts(artifacts, run_root)
        if args.drive_output_root:
            drive_root = (
                Path(args.drive_output_root).expanduser()
                / args.family
                / args.env
                / run_id
            )
            copied_drive = copy_selected_artifacts(artifacts, drive_root)

    run_config_out = run_root / "config.json"
    write_json(
        run_config_out,
        {
            "project_train_config": cfg,
            "resolved_training_settings": settings,
            "output_model_name": out_name,
            "run_id": run_id,
            "seed": args.seed,
            "mode": args.mode,
            "device": args.device,
        },
    )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "env": args.env,
        "mode": args.mode,
        "device": args.device,
        "seed": args.seed,
        "run_id": run_id,
        "output_model_name": out_name,
        "dry_run": args.dry_run,
        "launched_training": launched,
        "return_code": return_code,
        "command": cmd,
        "command_display": quote_cmd(cmd),
        "working_directory": str(swm_train_dir),
        "stablewm_home": str(stablewm_home),
        "pythonpath": env["PYTHONPATH"],
        "artifacts_discovered": artifacts,
        "copy_artifacts": args.copy_artifacts,
        "copied_local": copied_local,
        "copied_drive": copied_drive,
        "notes": cfg.get("notes"),
    }
    write_json(run_root / "metadata.json", metadata)

    print()
    print(f"Metadata written: {run_root / 'metadata.json'}")
    print(f"Resolved config written: {run_config_out}")
    if args.copy_artifacts:
        print(f"Copied local artifacts: {len(copied_local)}")
        if args.drive_output_root:
            print(f"Copied drive artifacts: {len(copied_drive)}")

    if launched and return_code not in (0, None):
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
