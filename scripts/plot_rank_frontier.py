from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method_category", "unknown"))
        grouped.setdefault(method, []).append(row)
    for method in grouped:
        grouped[method].sort(
            key=lambda r: float(r.get("predictor_compression_ratio"))
            if r.get("predictor_compression_ratio") is not None
            else -1.0
        )
    return grouped


def _plot_metric(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    y_key: str,
    y_label: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for method, rows in grouped.items():
        xs = []
        ys = []
        for row in rows:
            x = row.get("predictor_compression_ratio")
            y = row.get(y_key)
            if x is None or y is None:
                continue
            xs.append(float(x))
            ys.append(float(y))
        if xs:
            plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("Predictor Compression Ratio (new_params / old_params)")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-json",
        default="outputs/tables/rank_frontier_tworoom.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figures/rank_frontier_tworoom",
    )
    args = parser.parse_args()

    data = _read_json(Path(args.summary_json))
    rows = list(data.get("rows", []))
    grouped = _group(rows)
    out_dir = Path(args.output_dir)

    _plot_metric(
        grouped,
        y_key="operator_spearman_random",
        y_label="Random-cache Spearman",
        out_path=out_dir / "random_spearman_vs_predictor_ratio.png",
    )
    _plot_metric(
        grouped,
        y_key="operator_top5_random",
        y_label="Random-cache Top5 Overlap",
        out_path=out_dir / "random_top5_vs_predictor_ratio.png",
    )
    _plot_metric(
        grouped,
        y_key="operator_regret_random",
        y_label="Random-cache Teacher Regret",
        out_path=out_dir / "random_regret_vs_predictor_ratio.png",
    )
    _plot_metric(
        grouped,
        y_key="operator_spearman_dataset_action",
        y_label="Dataset-action Spearman",
        out_path=out_dir / "dataset_action_spearman_vs_predictor_ratio.png",
    )
    _plot_metric(
        grouped,
        y_key="closed_loop_success_rate",
        y_label="Closed-loop Success Rate",
        out_path=out_dir / "closed_loop_success_vs_predictor_ratio.png",
    )
    _plot_metric(
        grouped,
        y_key="compressed_total_params",
        y_label="Compressed Total Params",
        out_path=out_dir / "params_vs_predictor_ratio.png",
    )
    print(f"[done] wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
