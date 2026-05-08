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


def _metric_mean(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        mean = value.get("mean")
        return float(mean) if mean is not None else None
    if value is None:
        return None
    return float(value)


def _topk_mean(metrics: dict[str, Any], k: str) -> float | None:
    entry = metrics.get("topk_overlap", {}).get(k, {})
    mean = entry.get("mean")
    return float(mean) if mean is not None else None


def _base_tag(tag: str, eval_suffix: str) -> str:
    suffix = f"_{eval_suffix}"
    return tag[: -len(suffix)] if tag.endswith(suffix) else tag


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "eval_tag",
        "base_tag",
        "method_category",
        "rank_fraction",
        "predictor_compression_ratio",
        "spearman_mean",
        "top5_overlap_mean",
        "top10_overlap_mean",
        "teacher_regret_mean",
        "first_action_error_mean",
        "run_success",
        "notes_status",
    ]
    lines = [
        "# Held-out Operator Split Summary (TwoRoom)",
        "",
        "|" + "|".join(cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row.get(c)) for c in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_candidates(
    rows_by_base: dict[str, dict[str, Any]],
    teacher_checkpoint: str,
) -> list[dict[str, Any]]:
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
    candidates = [
        {
            "name": "teacher_lewm_hf_baseline",
            "tag": "teacher_lewm_hf",
            "model_path": teacher_checkpoint,
            "method": "teacher_baseline",
            "rank_fraction": 1.0,
            "predictor_compression_ratio": 1.0,
            "total_compression_ratio": 1.0,
            "heldout_operator_metrics": baseline_metrics,
            "rationale": (
                "Teacher baseline anchor used to define target operator."
            ),
            "status": "teacher",
        }
    ]

    spec = [
        (
            "lewm_tworoom_svd_r050_eval_s128_seed1",
            "Weight-SVD r=0.5",
            "candidate",
            "Naive low-rank baseline for moderate compression.",
        ),
        (
            "lewm_tworoom_aa_svd_r050_eval_s128_seed1",
            "AA-SVD r=0.5",
            "candidate",
            (
                "AA-SVD r=0.5 is the strongest compressed baseline "
                "by operator metrics."
            ),
        ),
        (
            "lewm_tworoom_svd_r050_cost_kl_split_eval_s128_seed1",
            "SVD r=0.5 + cost-KL split",
            "candidate",
            "Held-out cost-distribution operator recovery from naive SVD.",
        ),
        (
            "lewm_tworoom_svd_r050_elite_k10_split_eval_s128_seed1",
            "SVD r=0.5 + elite split",
            "candidate",
            "Held-out elite-set operator recovery from naive SVD.",
        ),
        (
            "lewm_tworoom_svd_r050_hybrid_split_eval_s128_seed1",
            "SVD r=0.5 + hybrid split",
            "candidate",
            (
                "Hybrid r=0.5 is the strongest operator-aware recovery from "
                "naive SVD r=0.5."
            ),
        ),
        (
            "lewm_tworoom_aa_svd_r025_eval_s128_seed1",
            "AA-SVD r=0.25",
            "candidate",
            "Aggressive non-operator-aware compression baseline.",
        ),
        (
            "lewm_tworoom_aa_svd_r025_hybrid_split_eval_s128_seed1",
            "AA-SVD r=0.25 + hybrid split",
            "candidate",
            (
                "AA-SVD r=0.25 + hybrid tests whether operator-aware recovery "
                "can improve stronger activation-aware compression."
            ),
        ),
    ]

    for eval_tag, name, status, rationale in spec:
        row = rows_by_base.get(eval_tag)
        if row is None:
            candidates.append(
                {
                    "name": name,
                    "tag": eval_tag,
                    "model_path": None,
                    "method": None,
                    "rank_fraction": None,
                    "predictor_compression_ratio": None,
                    "total_compression_ratio": None,
                    "heldout_operator_metrics": None,
                    "rationale": rationale,
                    "status": "failed",
                }
            )
            continue
        model_path = row.get("model_path")
        candidates.append(
            {
                "name": name,
                "tag": eval_tag,
                "model_path": model_path,
                "method": row.get("method_category"),
                "rank_fraction": row.get("rank_fraction"),
                "predictor_compression_ratio": row.get(
                    "predictor_compression_ratio"
                ),
                "total_compression_ratio": row.get("total_compression_ratio"),
                "heldout_operator_metrics": {
                    "finite_student_costs": row.get("finite_student_costs"),
                    "spearman_mean": row.get("spearman_mean"),
                    "top1_overlap_mean": row.get("top1_overlap_mean"),
                    "top5_overlap_mean": row.get("top5_overlap_mean"),
                    "top10_overlap_mean": row.get("top10_overlap_mean"),
                    "top20_overlap_mean": row.get("top20_overlap_mean"),
                    "teacher_regret_mean": row.get("teacher_regret_mean"),
                    "first_action_error_mean": row.get(
                        "first_action_error_mean"
                    ),
                    "teacher_best_index_match_rate": row.get(
                        "teacher_best_index_match_rate"
                    ),
                },
                "rationale": rationale,
                "status": status,
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom", choices=["tworoom"])
    parser.add_argument(
        "--eval-cache-tag",
        default="lewm_tworoom_eval_s128_c128_seed1",
    )
    parser.add_argument(
        "--eval-tag-suffix",
        default="eval_s128_seed1",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="quentinll/lewm-tworooms",
    )
    args = parser.parse_args()

    compression_root = Path("outputs/compression") / args.env
    metrics_root = Path("outputs/operator_metrics") / args.env
    out_root = Path("outputs/tables")
    out_root.mkdir(parents=True, exist_ok=True)

    compression_reports: dict[str, dict[str, Any]] = {}
    distill_reports: dict[str, dict[str, Any]] = {}
    for p in compression_root.glob("*/compression_report.json"):
        compression_reports[p.parent.name] = _read_json(p)
    for p in compression_root.glob("*/distillation_report.json"):
        distill_reports[p.parent.name] = _read_json(p)

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(metrics_root.glob("*/metrics.json")):
        metrics = _read_json(metrics_path)
        cache_path = str(metrics.get("cache_path", ""))
        if args.eval_cache_tag not in cache_path:
            continue
        eval_tag = str(metrics.get("tag", metrics_path.parent.name))
        base_tag = _base_tag(eval_tag, args.eval_tag_suffix)

        compression_report = compression_reports.get(base_tag)
        distill_report = distill_reports.get(base_tag)
        inherited = (
            distill_report.get("inherited_compression", {})
            if distill_report
            else {}
        )
        compression_method = (
            compression_report.get("method") if compression_report else None
        )
        distill_method = (
            distill_report.get("method") if distill_report else None
        )
        method_category = _method_category(compression_method, distill_method)
        run_success = (
            bool(distill_report.get("run_success"))
            if distill_report is not None
            else True
        )

        rank_fraction = inherited.get("rank_fraction")
        if rank_fraction is None and compression_report is not None:
            rank_fraction = compression_report.get("rank_fraction")

        predictor_ratio = inherited.get("predictor_compression_ratio")
        if predictor_ratio is None and compression_report is not None:
            predictor_ratio = compression_report.get(
                "predictor_compression_ratio"
            )

        total_ratio = inherited.get("total_compression_ratio")
        if total_ratio is None and compression_report is not None:
            total_ratio = compression_report.get("total_compression_ratio")

        if distill_report and distill_report.get("distilled_model_path"):
            model_path = distill_report.get("distilled_model_path")
        elif compression_report:
            model_path = compression_report.get("compressed_model_path")
        else:
            model_path = str(metrics.get("model_path", ""))

        notes = []
        if distill_report is not None:
            status = distill_report.get("method_status")
            if status:
                notes.append(f"status={status}")
            if distill_report.get("heldout_operator_validation") is True:
                notes.append("heldout=true")

        row = {
            "eval_tag": eval_tag,
            "base_tag": base_tag,
            "method_category": method_category,
            "init_base_method": (
                inherited.get("base_method") or compression_method
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
            "finite_student_costs": bool(metrics.get("finite_student_costs")),
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
                if metrics.get("teacher_best_index_match_rate") is not None
                else None
            ),
            "model_path": model_path,
            "notes_status": ";".join(notes),
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            str(r.get("method_category")),
            (
                9e9
                if r.get("rank_fraction") is None
                else float(r["rank_fraction"])
            ),
            str(r["eval_tag"]),
        )
    )

    csv_path = out_root / "operator_split_summary_tworoom.csv"
    json_path = out_root / "operator_split_summary_tworoom.json"
    md_path = out_root / "operator_split_summary_tworoom.md"
    candidate_path = out_root / "final_benchmark_candidates.json"

    fieldnames = [
        "eval_tag",
        "base_tag",
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
        "model_path",
        "notes_status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(
        json.dumps(
            {
                "env": args.env,
                "eval_cache_tag": args.eval_cache_tag,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(md_path, rows)

    rows_by_eval = {r["eval_tag"]: r for r in rows}
    candidates = _build_candidates(rows_by_eval, args.teacher_checkpoint)
    candidate_path.write_text(
        json.dumps(
            {
                "env": args.env,
                "eval_cache_tag": args.eval_cache_tag,
                "candidates": candidates,
                "interpretation": [
                    "AA-SVD r=0.5 is the strongest compressed baseline by "
                    "operator metrics.",
                    "Hybrid r=0.5 is the strongest operator-aware recovery "
                    "from naive SVD r=0.5.",
                    (
                        "AA-SVD r=0.25 + hybrid tests whether operator-aware "
                        "recovery can improve stronger activation-aware "
                        "compression."
                    ),
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Held-out split operator summary written")
    print(f"  csv:        {csv_path}")
    print(f"  json:       {json_path}")
    print(f"  markdown:   {md_path}")
    print(f"  candidates: {candidate_path}")


if __name__ == "__main__":
    main()
