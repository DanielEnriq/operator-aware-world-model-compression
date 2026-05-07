from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import stable_worldmodel as swm


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def stablewm_home() -> Path:
    return Path(
        os.environ.get("STABLEWM_HOME", project_root() / ".swm_cache")
    ).expanduser()


def bytes_to_str(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size}B"


def discover_runs(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    if not root.exists():
        return []

    patterns = (
        "**/config.json",
        "**/config.yaml",
        "**/weights_epoch_*.pt",
        "**/*_object.ckpt",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            candidates.add(path.parent)

    return sorted(candidates)


def inspect_run(run_dir: Path) -> dict[str, Any]:
    config_json = run_dir / "config.json"
    config_yaml = run_dir / "config.yaml"
    pt_files = sorted(run_dir.glob("weights_epoch_*.pt"))
    object_ckpts = sorted(run_dir.glob("*_object.ckpt"))

    files = []
    for p in [config_json, config_yaml, *pt_files, *object_ckpts]:
        if p.exists():
            files.append(
                {
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "size_human": bytes_to_str(p.stat().st_size),
                }
            )

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "config_json": str(config_json) if config_json.exists() else None,
        "config_yaml": str(config_yaml) if config_yaml.exists() else None,
        "weights_pt": [str(p) for p in pt_files],
        "object_ckpts": [str(p) for p in object_ckpts],
        "files": files,
        "load_attempts": [],
    }

    load_pretrained_name = str(run_dir)
    try:
        model = swm.wm.utils.load_pretrained(load_pretrained_name)
        report["load_attempts"].append(
            {
                "loader": "stable_worldmodel.wm.utils.load_pretrained",
                "target": load_pretrained_name,
                "ok": True,
                "model_type": str(type(model)),
                "has_get_cost": hasattr(model, "get_cost"),
            }
        )
    except Exception as exc:
        report["load_attempts"].append(
            {
                "loader": "stable_worldmodel.wm.utils.load_pretrained",
                "target": load_pretrained_name,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    auto_target = str(run_dir)
    try:
        model = swm.policy.AutoCostModel(auto_target)
        report["load_attempts"].append(
            {
                "loader": "stable_worldmodel.policy.AutoCostModel",
                "target": auto_target,
                "ok": True,
                "model_type": str(type(model)),
                "has_get_cost": hasattr(model, "get_cost"),
            }
        )
    except Exception as exc:
        report["load_attempts"].append(
            {
                "loader": "stable_worldmodel.policy.AutoCostModel",
                "target": auto_target,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Root directory to scan. Default scans both "
            "$STABLEWM_HOME/checkpoints and project checkpoints/."
        ),
    )
    args = parser.parse_args()

    roots: list[Path]
    if args.root:
        roots = [Path(args.root).expanduser()]
    else:
        roots = [
            stablewm_home() / "checkpoints",
            project_root() / "checkpoints",
        ]

    all_runs: list[Path] = []
    for root in roots:
        runs = discover_runs(root)
        print(f"\nScan root: {root}")
        print(f"Candidate runs: {len(runs)}")
        all_runs.extend(runs)

    dedup_runs = sorted({p.resolve() for p in all_runs})
    print(f"\nTotal unique candidate runs: {len(dedup_runs)}")

    run_reports: list[dict[str, Any]] = []
    for run_dir in dedup_runs:
        print("\n" + "=" * 80)
        print(f"Run: {run_dir}")
        report = inspect_run(run_dir)
        for file_info in report["files"]:
            print(f"  file: {file_info['path']} ({file_info['size_human']})")
        for attempt in report["load_attempts"]:
            status = "OK" if attempt["ok"] else "FAIL"
            print(f"  {status:4s} {attempt['loader']}")
            if not attempt["ok"]:
                print(f"       {attempt['error']}")
            else:
                print(f"       has_get_cost={attempt['has_get_cost']}")
        run_reports.append(report)

    output_dir = project_root() / "outputs" / "inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"checkpoint_inspection_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "roots": [str(r) for r in roots],
                "run_count": len(run_reports),
                "runs": run_reports,
            },
            f,
            indent=2,
        )

    print(f"\nInspection report written: {out_path}")


if __name__ == "__main__":
    main()
