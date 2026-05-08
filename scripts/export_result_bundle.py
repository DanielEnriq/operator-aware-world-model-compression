from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--output-root", default="outputs/result_bundles")
    args = parser.parse_args()

    bundle_root = Path(args.output_root) / args.run_id
    tables_dir = bundle_root / "tables"
    figures_dir = bundle_root / "figures"
    benches_dir = bundle_root / "benchmarks"
    comp_dir = bundle_root / "compression_reports"
    distill_dir = bundle_root / "distill_reports"
    for p in [tables_dir, figures_dir, benches_dir, comp_dir, distill_dir]:
        p.mkdir(parents=True, exist_ok=True)

    table_base = (
        Path("outputs/tables")
        / f"full_closed_loop_frontier_{args.env}_{args.model_family}"
    )
    summary_json = table_base.with_suffix(".json")
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing summary json: {summary_json}")
    rows = list(_read_json(summary_json).get("rows", []))

    artifacts: list[dict[str, str]] = []
    for ext in [".csv", ".json", ".md"]:
        src = table_base.with_suffix(ext)
        dst = tables_dir / src.name
        if _copy_if_exists(src, dst):
            artifacts.append({"role": "table", "path": str(dst)})

    fig_png = Path("outputs/figures") / f"full_closed_loop_frontier_{args.env}_{args.model_family}.png"
    fig_pdf = Path("outputs/figures") / f"full_closed_loop_frontier_{args.env}_{args.model_family}.pdf"
    for src in [fig_png, fig_pdf]:
        dst = figures_dir / src.name
        if _copy_if_exists(src, dst):
            artifacts.append({"role": "figure", "path": str(dst)})

    for row in rows:
        bench = row.get("benchmark_json")
        if isinstance(bench, str):
            src = Path(bench)
            dst = benches_dir / src.name
            if _copy_if_exists(src, dst):
                artifacts.append({"role": "benchmark", "path": str(dst)})
        comp = row.get("compression_report")
        if isinstance(comp, str):
            src = Path(comp)
            dst = comp_dir / src.parent.name / src.name
            if _copy_if_exists(src, dst):
                artifacts.append({"role": "compression_report", "path": str(dst)})
        distill = row.get("distill_report")
        if isinstance(distill, str):
            src = Path(distill)
            dst = distill_dir / src.parent.name / src.name
            if _copy_if_exists(src, dst):
                artifacts.append({"role": "distill_report", "path": str(dst)})

    manifest = {
        "schema_version": 1,
        "kind": "result_bundle",
        "run_id": args.run_id,
        "env": args.env,
        "model_family": args.model_family,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows_count": len(rows),
        "artifacts": artifacts,
    }
    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {manifest_path}")


if __name__ == "__main__":
    main()
