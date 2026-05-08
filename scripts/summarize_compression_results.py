from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_category(
    compression_method: str | None,
    distill_method: str | None,
) -> str:
    if distill_method == "operator_cost_kl_distillation":
        return "cost_kl"
    if distill_method == "operator_elite_distillation":
        return "elite"
    if distill_method == "operator_hybrid_distillation":
        return "hybrid"
    if distill_method == "prediction_only_distillation":
        return "prediction_only_distill"
    if compression_method == "weight_svd":
        return "weight_svd"
    if compression_method == "activation_aware_svd":
        return "activation_aware_svd"
    return "unknown"


def _metric_mean(
    metrics: dict[str, Any] | None,
    key: str,
) -> float | None:
    if metrics is None:
        return None
    value = metrics.get(key)
    if isinstance(value, dict):
        mean = value.get("mean")
        return float(mean) if mean is not None else None
    if value is None:
        return None
    return float(value)


def _topk_mean(metrics: dict[str, Any] | None, k: str) -> float | None:
    if metrics is None:
        return None
    topk = metrics.get("topk_overlap", {})
    item = topk.get(k, {})
    mean = item.get("mean")
    return float(mean) if mean is not None else None


def _first_non_none(values: list[Any]) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _build_row(
    tag: str,
    compression_report: dict[str, Any] | None,
    distill_report: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    compression_method = (
        compression_report.get("method") if compression_report else None
    )
    distill_method = distill_report.get("method") if distill_report else None
    inherited = (
        distill_report.get("inherited_compression", {})
        if distill_report
        else {}
    )
    method_category = _method_category(compression_method, distill_method)

    rank_fraction = _first_non_none(
        [
            inherited.get("rank_fraction"),
            (
                compression_report.get("rank_fraction")
                if compression_report
                else None
            ),
        ]
    )
    predictor_ratio = _first_non_none(
        [
            inherited.get("predictor_compression_ratio"),
            compression_report.get("predictor_compression_ratio")
            if compression_report
            else None,
        ]
    )
    total_ratio = _first_non_none(
        [
            inherited.get("total_compression_ratio"),
            compression_report.get("total_compression_ratio")
            if compression_report
            else None,
        ]
    )

    run_success = (
        bool(distill_report.get("run_success"))
        if distill_report is not None
        else True
    )
    finite_costs = (
        bool(metrics.get("finite_student_costs"))
        if (
            metrics is not None
            and metrics.get("finite_student_costs") is not None
        )
        else None
    )
    notes = []
    if distill_report is not None:
        method_status = distill_report.get("method_status")
        if method_status:
            notes.append(f"status={method_status}")
    if metrics is None:
        notes.append("missing_operator_metrics")

    initial_loss = _first_non_none(
        [
            (
                distill_report.get("initial_total_loss")
                if distill_report
                else None
            ),
            distill_report.get("initial_train_kl") if distill_report else None,
            distill_report.get("initial_train_elite_loss")
            if distill_report
            else None,
            (
                distill_report.get("initial_train_loss")
                if distill_report
                else None
            ),
        ]
    )
    final_loss = _first_non_none(
        [
            distill_report.get("final_total_loss") if distill_report else None,
            distill_report.get("final_train_kl") if distill_report else None,
            distill_report.get("final_train_elite_loss")
            if distill_report
            else None,
            distill_report.get("final_train_loss") if distill_report else None,
        ]
    )

    row = {
        "tag": tag,
        "method_category": method_category,
        "init_base_method": _first_non_none(
            [
                inherited.get("base_method"),
                compression_method,
            ]
        ),
        "rank_fraction": (
            float(rank_fraction) if rank_fraction is not None else None
        ),
        "predictor_compression_ratio": (
            float(predictor_ratio) if predictor_ratio is not None else None
        ),
        "total_compression_ratio": (
            float(total_ratio) if total_ratio is not None else None
        ),
        "run_success": run_success,
        "finite_student_costs": finite_costs,
        "spearman_mean": _metric_mean(metrics, "spearman_per_state"),
        "top1_overlap_mean": _topk_mean(metrics, "1"),
        "top5_overlap_mean": _topk_mean(metrics, "5"),
        "top10_overlap_mean": _topk_mean(metrics, "10"),
        "top20_overlap_mean": _topk_mean(metrics, "20"),
        "teacher_regret_mean": _metric_mean(metrics, "teacher_regret"),
        "first_action_error_mean": _metric_mean(
            metrics,
            "selected_first_action_error",
        ),
        "teacher_best_index_match_rate": (
            float(metrics.get("teacher_best_index_match_rate"))
            if (
                metrics
                and metrics.get("teacher_best_index_match_rate") is not None
            )
            else None
        ),
        "initial_loss": (
            float(initial_loss) if initial_loss is not None else None
        ),
        "final_loss": float(final_loss) if final_loss is not None else None,
        "notes_status": ";".join(notes),
    }
    return row


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "tag",
        "method_category",
        "rank_fraction",
        "predictor_compression_ratio",
        "run_success",
        "spearman_mean",
        "top5_overlap_mean",
        "top10_overlap_mean",
        "teacher_regret_mean",
        "first_action_error_mean",
        "notes_status",
    ]
    lines = [
        "# Compression Summary (TwoRoom)",
        "",
        "|" + "|".join(cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append(
            "|" + "|".join(_fmt(row.get(c)) for c in cols) + "|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_candidates(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    def _pick(tag: str) -> dict[str, Any] | None:
        return rows.get(tag)

    candidates = []
    baseline_metrics = {
        "spearman_mean": 1.0,
        "top1_overlap_mean": 1.0,
        "top5_overlap_mean": 1.0,
        "top10_overlap_mean": 1.0,
        "teacher_regret_mean": 0.0,
        "first_action_error_mean": 0.0,
        "teacher_best_index_match_rate": 1.0,
        "source": "self-reference teacher cache ideal",
    }
    candidates.append(
        {
            "name": "teacher_lewm_hf_baseline",
            "model_path": "quentinll/lewm-tworooms",
            "method": "teacher_baseline",
            "rank_fraction": 1.0,
            "predictor_compression_ratio": 1.0,
            "operator_metrics": baseline_metrics,
            "rationale": (
                "Teacher reference used to define operator targets; "
                "serves as uncompressed benchmark anchor."
            ),
        }
    )

    wanted = [
        (
            "lewm_tworoom_svd_r050",
            "Weight-SVD r=0.5",
            "Naive low-rank baseline for moderate compression.",
        ),
        (
            "lewm_tworoom_aa_svd_r050",
            "AA-SVD r=0.5",
            "Strongest compressed baseline by operator metrics.",
        ),
        (
            "lewm_tworoom_svd_r050_hybrid",
            "SVD r=0.5 + hybrid",
            "Strongest operator-aware recovery from naive SVD r=0.5.",
        ),
        (
            "lewm_tworoom_aa_svd_r025",
            "AA-SVD r=0.25",
            "Aggressive non-operator-aware compression baseline.",
        ),
        (
            "lewm_tworoom_aa_svd_r025_hybrid",
            "AA-SVD r=0.25 + hybrid",
            (
                "Tests operator-aware recovery on stronger "
                "activation-aware model."
            ),
        ),
        (
            "lewm_tworoom_svd_r025_hybrid",
            "SVD r=0.25 + hybrid",
            "Optional high-compression stress candidate.",
        ),
    ]

    for tag, label, rationale in wanted:
        row = _pick(tag)
        if row is None:
            continue
        candidates.append(
            {
                "name": label,
                "tag": tag,
                "model_path": (
                    f"outputs/compression/tworoom/{tag}/distilled_model.pt"
                )
                if "hybrid" in tag
                else (
                    f"outputs/compression/tworoom/{tag}/compressed_model.pt"
                ),
                "method": row.get("method_category"),
                "rank_fraction": row.get("rank_fraction"),
                "predictor_compression_ratio": row.get(
                    "predictor_compression_ratio"
                ),
                "total_compression_ratio": row.get("total_compression_ratio"),
                "operator_metrics": {
                    "finite_student_costs": row.get("finite_student_costs"),
                    "spearman_mean": row.get("spearman_mean"),
                    "top1_overlap_mean": row.get("top1_overlap_mean"),
                    "top5_overlap_mean": row.get("top5_overlap_mean"),
                    "top10_overlap_mean": row.get("top10_overlap_mean"),
                    "teacher_regret_mean": row.get("teacher_regret_mean"),
                    "first_action_error_mean": row.get(
                        "first_action_error_mean"
                    ),
                    "teacher_best_index_match_rate": row.get(
                        "teacher_best_index_match_rate"
                    ),
                },
                "rationale": rationale,
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="tworoom",
        choices=["tworoom"],
    )
    args = parser.parse_args()

    compression_root = Path("outputs/compression") / args.env
    metrics_root = Path("outputs/operator_metrics") / args.env
    out_root = Path("outputs/tables")
    out_root.mkdir(parents=True, exist_ok=True)

    compression_reports: dict[str, dict[str, Any]] = {}
    distill_reports: dict[str, dict[str, Any]] = {}
    metrics_reports: dict[str, dict[str, Any]] = {}

    for path in compression_root.glob("*/compression_report.json"):
        compression_reports[path.parent.name] = _read_json(path)
    for path in compression_root.glob("*/distillation_report.json"):
        distill_reports[path.parent.name] = _read_json(path)
    for path in metrics_root.glob("*/metrics.json"):
        metrics_reports[path.parent.name] = _read_json(path)

    tags = sorted(
        set(compression_reports.keys())
        | set(distill_reports.keys())
        | set(metrics_reports.keys())
    )

    rows = [
        _build_row(
            tag=tag,
            compression_report=compression_reports.get(tag),
            distill_report=distill_reports.get(tag),
            metrics=metrics_reports.get(tag),
        )
        for tag in tags
    ]
    rows.sort(
        key=lambda r: (
            str(r.get("method_category")),
            (
                9e9
                if r.get("rank_fraction") is None
                else float(r["rank_fraction"])
            ),
            str(r["tag"]),
        )
    )

    csv_path = out_root / "compression_summary_tworoom.csv"
    json_path = out_root / "compression_summary_tworoom.json"
    md_path = out_root / "compression_summary_tworoom.md"
    candidates_path = out_root / "final_benchmark_candidates.json"

    fields = [
        "tag",
        "method_category",
        "init_base_method",
        "rank_fraction",
        "predictor_compression_ratio",
        "total_compression_ratio",
        "run_success",
        "finite_student_costs",
        "spearman_mean",
        "top1_overlap_mean",
        "top5_overlap_mean",
        "top10_overlap_mean",
        "top20_overlap_mean",
        "teacher_regret_mean",
        "first_action_error_mean",
        "teacher_best_index_match_rate",
        "initial_loss",
        "final_loss",
        "notes_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(
        json.dumps({"env": args.env, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(md_path, rows)

    row_map = {r["tag"]: r for r in rows}
    candidates = _make_candidates(row_map)
    candidates_path.write_text(
        json.dumps(
            {
                "env": args.env,
                "candidates": candidates,
                "interpretation": [
                    "AA-SVD r=0.5 is the strongest compressed baseline by "
                    "operator metrics.",
                    "Hybrid r=0.5 is the strongest operator-aware recovery "
                    "from naive SVD r=0.5.",
                    "AA-SVD r=0.25 + hybrid tests whether operator-aware "
                    "recovery improves a stronger activation-aware compressed "
                    "model at aggressive rank.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Compression summary written")
    print(f"  csv:        {csv_path}")
    print(f"  json:       {json_path}")
    print(f"  markdown:   {md_path}")
    print(f"  candidates: {candidates_path}")


if __name__ == "__main__":
    main()
