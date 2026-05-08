from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

METHOD_LABELS = {
    "teacher": "Teacher",
    "weight_svd": "Weight SVD",
    "activation_svd": "Activation-aware SVD",
    "operator_cost_kl": "Operator-aware KL",
    "operator_hybrid": "Operator-aware Hybrid",
    "operator_elite": "Operator-aware Elite",
}
METHOD_ORDER = {
    "teacher": 0,
    "weight_svd": 1,
    "activation_svd": 2,
    "operator_cost_kl": 3,
    "operator_hybrid": 4,
    "operator_elite": 5,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method"))
        grouped.setdefault(method, []).append(row)
    for method, method_rows in grouped.items():
        method_rows.sort(
            key=lambda row: float(row.get("total_compression_ratio", 1.0))
            if isinstance(row.get("total_compression_ratio"), (int, float))
            else 1.0
        )
    return dict(
        sorted(
            grouped.items(),
            key=lambda kv: METHOD_ORDER.get(kv[0], 99),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--num-eval", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--x-metric",
        default="total_compression_ratio",
        choices=["total_compression_ratio", "predictor_compression_ratio"],
    )
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--write-pdf", action="store_true")
    args = parser.parse_args()

    payload = _read_json(Path(args.summary_json))
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("status") in {"ok", "skipped_existing"}
    ]
    grouped = _group(rows)

    fig_name = f"full_closed_loop_frontier_{args.env}_{args.model_family}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{fig_name}.png"
    pdf_path = out_dir / f"{fig_name}.pdf"

    plt.figure(figsize=(8, 5))
    teacher_y = None
    for method, method_rows in grouped.items():
        label = METHOD_LABELS.get(method, method)
        if method == "teacher":
            for row in method_rows:
                if isinstance(row.get("success_rate"), (int, float)):
                    teacher_y = float(row["success_rate"])
                    x = (
                        float(row.get(args.x_metric, 1.0))
                        if isinstance(row.get(args.x_metric), (int, float))
                        else 1.0
                    )
                    plt.scatter([x], [teacher_y], label=label, marker="*", s=160)
            continue

        xs: list[float] = []
        ys: list[float] = []
        for row in method_rows:
            x = row.get(args.x_metric)
            y = row.get("success_rate")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                xs.append(float(x))
                ys.append(float(y))
        if xs:
            plt.plot(xs, ys, marker="o", label=label)

    if teacher_y is not None:
        plt.axhline(
            y=teacher_y,
            linestyle="--",
            linewidth=1.0,
            label="Teacher anchor",
        )
    plt.xlabel(
        "Total Compression Ratio (new params / old params)"
        if args.x_metric == "total_compression_ratio"
        else "Predictor Compression Ratio (new params / old params)"
    )
    plt.ylabel("Closed-loop MPC success rate")
    plt.title(
        f"{args.env} {args.model_family} frontier "
        f"(n={args.num_eval}, seed={args.seed}; preliminary if n=64)"
    )
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    if args.write_pdf:
        plt.savefig(pdf_path)
    plt.close()
    print(f"[done] wrote {png_path}")
    if args.write_pdf:
        print(f"[done] wrote {pdf_path}")


if __name__ == "__main__":
    main()
