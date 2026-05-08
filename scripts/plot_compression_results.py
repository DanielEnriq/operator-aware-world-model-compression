from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid summary JSON at {path}")
    return rows


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _plot_metric(
    rows: list[dict[str, Any]],
    *,
    y_key: str,
    y_label: str,
    filename: str,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped: dict[str, list[tuple[float, float, str]]] = {}
    for row in rows:
        x = _to_float(row.get("predictor_compression_ratio"))
        y = _to_float(row.get(y_key))
        if x is None or y is None:
            continue
        method = str(row.get("method_category", "unknown"))
        grouped.setdefault(method, []).append((x, y, str(row.get("tag", ""))))

    if not grouped:
        ax.text(
            0.5,
            0.5,
            "No data points available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        for idx, (method, points) in enumerate(sorted(grouped.items())):
            points_sorted = sorted(points, key=lambda p: p[0])
            xs = [p[0] for p in points_sorted]
            ys = [p[1] for p in points_sorted]
            ax.plot(
                xs,
                ys,
                marker=markers[idx % len(markers)],
                linewidth=1.5,
                label=method,
            )

    ax.set_xlabel("Predictor Compression Ratio")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs Predictor Compression")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-json",
        default="outputs/tables/compression_summary_tworoom.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figures/compression",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(summary_path)
    _plot_metric(
        rows,
        y_key="teacher_regret_mean",
        y_label="Teacher Regret (mean)",
        filename="regret_vs_predictor_compression.png",
        out_dir=out_dir,
    )
    _plot_metric(
        rows,
        y_key="top5_overlap_mean",
        y_label="Top-5 Overlap (mean)",
        filename="top5_overlap_vs_predictor_compression.png",
        out_dir=out_dir,
    )
    _plot_metric(
        rows,
        y_key="spearman_mean",
        y_label="Spearman (mean)",
        filename="spearman_vs_predictor_compression.png",
        out_dir=out_dir,
    )
    _plot_metric(
        rows,
        y_key="first_action_error_mean",
        y_label="First-Action Error (mean)",
        filename="first_action_error_vs_predictor_compression.png",
        out_dir=out_dir,
    )

    print("Compression figures written")
    print(f"  output_dir: {out_dir}")


if __name__ == "__main__":
    main()
