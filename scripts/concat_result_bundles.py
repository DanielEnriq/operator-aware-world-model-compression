from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

TABLE_COLUMNS = [
    "env",
    "model_family",
    "teacher_checkpoint",
    "base_checkpoint_or_path",
    "method",
    "init_method",
    "rank_fraction",
    "predictor_compression_ratio",
    "total_compression_ratio",
    "original_total_params",
    "compressed_total_params",
    "original_predictor_params",
    "compressed_predictor_params",
    "layers_compressed",
    "compression_status",
    "distill_steps",
    "distill_batch_size",
    "distill_train_cache",
    "distill_loss",
    "distill_wall_time_s",
    "num_eval",
    "seed",
    "success_rate",
    "num_successes",
    "avg_return",
    "avg_final_distance",
    "eval_time_s",
    "model_path",
    "benchmark_json",
    "compression_report",
    "distill_report",
    "status",
    "error_message",
    "notes",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _row_id(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("env")),
        str(row.get("model_family")),
        str(row.get("method")),
        str(row.get("rank_fraction")),
        str(row.get("model_path")),
    )


def _plot(rows: list[dict[str, Any]], out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('env')}:{row.get('model_family')}:{row.get('method')}"
        grouped.setdefault(key, []).append(row)
    for key, group in grouped.items():
        xs: list[float] = []
        ys: list[float] = []
        for row in group:
            x = row.get("total_compression_ratio")
            y = row.get("success_rate")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                xs.append(float(x))
                ys.append(float(y))
        if xs:
            plt.plot(xs, ys, marker="o", label=key)
    plt.xlabel("Total Compression Ratio (new params / old params)")
    plt.ylabel("Closed-loop success rate")
    plt.title("Combined Frontiers Across Bundles")
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-roots", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Combined CSV path.")
    args = parser.parse_args()

    rows_all: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for bundle_root in [Path(p) for p in args.bundle_roots]:
        manifest_path = bundle_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        tables_dir = bundle_root / "tables"
        summary_jsons = sorted(tables_dir.glob("full_closed_loop_frontier_*.json"))
        if not summary_jsons:
            continue
        for summary_json in summary_jsons:
            payload = _read_json(summary_json)
            for row in payload.get("rows", []):
                row_key = _row_id(row)
                if row_key in seen:
                    continue
                seen.add(row_key)
                rows_all.append(row)

    out_csv = Path(args.output)
    out_json = out_csv.with_suffix(".json")
    out_md = out_csv.with_suffix(".md")
    out_png = Path("outputs/figures/all_frontiers_combined.png")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows_all:
            writer.writerow({key: row.get(key) for key in TABLE_COLUMNS})

    out_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "combined_full_closed_loop_frontier",
                "rows": rows_all,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Combined Full Frontier",
        "",
        "Closed-loop success is the primary metric.",
        "",
        "|" + "|".join(TABLE_COLUMNS) + "|",
        "|" + "|".join(["---"] * len(TABLE_COLUMNS)) + "|",
    ]
    for row in rows_all:
        lines.append("|" + "|".join(_fmt(row.get(c)) for c in TABLE_COLUMNS) + "|")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _plot(rows_all, out_png)
    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {out_json}")
    print(f"[done] wrote {out_md}")
    print(f"[done] wrote {out_png}")


if __name__ == "__main__":
    main()
