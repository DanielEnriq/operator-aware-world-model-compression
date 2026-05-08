from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Invalid summary format: {path}")
    return rows


def _plot_metric(
    rows: list[dict[str, Any]],
    *,
    y_key: str,
    y_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        x = _to_float(row.get("predictor_compression_ratio"))
        y = _to_float(row.get(y_key))
        if x is None or y is None:
            continue
        method = str(row.get("method_category", "unknown"))
        grouped.setdefault(method, []).append((x, y))

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for idx, (method, points) in enumerate(sorted(grouped.items())):
        points = sorted(points, key=lambda t: t[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            marker=markers[idx % len(markers)],
            linewidth=1.5,
            label=method,
        )

    if not grouped:
        ax.text(
            0.5,
            0.5,
            "No held-out rows available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel("Predictor Compression Ratio")
    ax.set_ylabel(y_label)
    ax.set_title(f"Held-out {y_label} vs Predictor Compression")
    ax.grid(True, alpha=0.3)
    if grouped:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-json",
        default="outputs/tables/operator_split_summary_tworoom.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figures/compression_split",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(summary_path)

    _plot_metric(
        rows,
        y_key="teacher_regret_mean",
        y_label="Regret (mean)",
        out_path=out_dir / "heldout_regret_vs_predictor_compression.png",
    )
    _plot_metric(
        rows,
        y_key="top5_overlap_mean",
        y_label="Top-5 Overlap (mean)",
        out_path=out_dir / "heldout_top5_overlap_vs_predictor_compression.png",
    )
    _plot_metric(
        rows,
        y_key="spearman_mean",
        y_label="Spearman (mean)",
        out_path=out_dir / "heldout_spearman_vs_predictor_compression.png",
    )
    _plot_metric(
        rows,
        y_key="first_action_error_mean",
        y_label="First-Action Error (mean)",
        out_path=(
            out_dir
            / "heldout_first_action_error_vs_predictor_compression.png"
        ),
    )

    print("Held-out split figures written")
    print(f"  output_dir: {out_dir}")


if __name__ == "__main__":
    main()
