from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _base_tag_from_eval_tag(tag: str) -> str:
    for suffix in [
        "_eval_s128_seed1",
        "_eval_dataset_actions_s128_seed2",
        "_eval_near_elite_s128_seed3",
    ]:
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


def _method_category(tag: str) -> str:
    if "_aa_svd_" in tag:
        return "activation_aware_svd"
    if "_svd_" in tag and "_cost_kl" not in tag and "_hybrid" not in tag:
        return "weight_svd"
    if "_cost_kl" in tag:
        return "cost_kl"
    if "_hybrid" in tag:
        return "hybrid"
    if "teacher" in tag:
        return "teacher"
    return "unknown"


def _read_metric(path: Path, metric_key: str) -> float | None:
    data = _read_json(path)
    value = data.get(metric_key)
    if isinstance(value, dict):
        mean = value.get("mean")
        return float(mean) if mean is not None else None
    if value is None:
        return None
    return float(value)


def _metric_from_payload(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, dict):
        mean = value.get("mean")
        return float(mean) if mean is not None else None
    if value is None:
        return None
    return float(value)


def _candidate_mode_for_metrics(metrics: dict[str, Any]) -> str:
    cache_path = metrics.get("cache_path")
    if cache_path is None:
        return "unknown"
    cpath = Path(str(cache_path))
    if not cpath.exists():
        return "unknown"
    cache = _read_json(cpath) if cpath.suffix == ".json" else None
    if cache is not None:
        return str(cache.get("candidate_mode", "unknown"))
    try:
        import torch

        payload = torch.load(cpath, map_location="cpu", weights_only=False)
        return str(payload.get("candidate_mode", "random"))
    except Exception:
        return "unknown"


def _collect_closed_loop(bench_root: Path) -> dict[str, float]:
    by_tag: dict[str, float] = {}
    for p in bench_root.glob("*/*_seed*_n*.json"):
        data = _read_json(p)
        model = data.get("model", {})
        model_path = model.get("model_path")
        if model_path:
            base = Path(model_path).parent.name
        else:
            name = str(model.get("name", ""))
            base = name.split("_closed_loop_seed")[0]
        by_tag[base] = float(
            data.get("performance", {}).get("success_rate", 0.0)
        )
    return by_tag


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    args = parser.parse_args()

    compression_root = Path("outputs/compression") / args.env
    metrics_root = Path("outputs/operator_metrics") / args.env
    bench_root = Path("outputs/benchmarks") / args.env
    pred_root = Path("outputs/prediction_rollout") / args.env
    out_root = Path("outputs/tables")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "rank_frontier_run_manifest_tworoom.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    identity_ok = manifest.get("identity_check_passed")
    teacher_anchor_source = manifest.get("teacher_anchor_source")

    compression_reports: dict[str, dict[str, Any]] = {}
    for p in compression_root.glob("*/compression_report.json"):
        compression_reports[p.parent.name] = _read_json(p)

    random_eval: dict[str, dict[str, Any]] = {}
    dataset_eval: dict[str, dict[str, Any]] = {}
    near_elite_eval: dict[str, dict[str, Any]] = {}
    for p in metrics_root.glob("*/metrics.json"):
        metrics = _read_json(p)
        base_tag = _base_tag_from_eval_tag(
            str(metrics.get("tag", p.parent.name))
        )
        cache_path = str(metrics.get("cache_path", ""))
        mode = "random"
        if "dataset_actions" in cache_path:
            mode = "dataset_actions"
        elif "near_elite" in cache_path:
            mode = "near_elite"
        elif "cem" in cache_path:
            mode = "cem"
        if mode == "dataset_actions":
            dataset_eval[base_tag] = metrics
        elif mode in {"near_elite", "cem"}:
            near_elite_eval[base_tag] = metrics
        else:
            random_eval[base_tag] = metrics

    pred_eval: dict[str, float] = {}
    for p in pred_root.glob("*/metrics.json"):
        data = _read_json(p)
        pred_eval[str(data.get("model_tag", p.parent.name))] = float(
            data.get("teacher_student_cost_mse", 0.0)
        )

    closed_loop = _collect_closed_loop(bench_root)
    tags = sorted(set(compression_reports) | set(random_eval) | set(closed_loop))
    rows: list[dict[str, Any]] = []
    for tag in tags:
        report = compression_reports.get(tag, {})
        random_metrics = random_eval.get(tag)
        dataset_metrics = dataset_eval.get(tag)
        near_metrics = near_elite_eval.get(tag)
        notes: list[str] = []
        if random_metrics is None:
            notes.append("missing_random_eval")
        if dataset_metrics is None:
            notes.append("missing_dataset_action_eval")
        if near_metrics is None:
            notes.append("missing_near_elite_eval")
        row = {
            "tag": tag,
            "method_category": _method_category(tag),
            "rank_fraction": report.get("rank_fraction"),
            "predictor_compression_ratio": report.get(
                "predictor_compression_ratio"
            ),
            "total_compression_ratio": report.get("total_compression_ratio"),
            "original_predictor_params": report.get("predictor_params_before"),
            "compressed_predictor_params": report.get("predictor_params_after"),
            "original_total_params": report.get("total_params_before"),
            "compressed_total_params": report.get("total_params_after"),
            "model_path": report.get("compressed_model_path"),
            "compression_report_path": (
                str(compression_root / tag / "compression_report.json")
                if (compression_root / tag / "compression_report.json").exists()
                else None
            ),
            "operator_spearman_random": (
                _metric_from_payload(random_metrics, "spearman_per_state")
                if random_metrics
                else None
            ),
            "operator_top5_random": (
                random_metrics.get("topk_overlap", {}).get("5", {}).get("mean")
                if random_metrics
                else None
            ),
            "operator_top10_random": (
                random_metrics.get("topk_overlap", {})
                .get("10", {})
                .get("mean")
                if random_metrics
                else None
            ),
            "operator_regret_random": (
                _metric_from_payload(random_metrics, "teacher_regret")
                if random_metrics
                else None
            ),
            "first_action_error_random": (
                _metric_from_payload(random_metrics, "selected_first_action_error")
                if random_metrics
                else None
            ),
            "raw_cost_mse_random": (
                float(random_metrics.get("raw_cost_mse"))
                if (
                    random_metrics
                    and random_metrics.get("raw_cost_mse") is not None
                )
                else None
            ),
            "operator_spearman_dataset_action": (
                _metric_from_payload(dataset_metrics, "spearman_per_state")
                if dataset_metrics
                else None
            ),
            "operator_spearman_near_elite": (
                _metric_from_payload(near_metrics, "spearman_per_state")
                if near_metrics
                else None
            ),
            "closed_loop_success_rate": closed_loop.get(tag),
            "prediction_rollout_error": pred_eval.get(tag),
            "random_operator_eval_path": (
                str(metrics_root / f"{tag}_eval_s128_seed1" / "metrics.json")
                if (
                    metrics_root / f"{tag}_eval_s128_seed1" / "metrics.json"
                ).exists()
                else None
            ),
            "dataset_action_operator_eval_path": (
                str(
                    metrics_root
                    / f"{tag}_eval_dataset_actions_s128_seed2"
                    / "metrics.json"
                )
                if (
                    metrics_root
                    / f"{tag}_eval_dataset_actions_s128_seed2"
                    / "metrics.json"
                ).exists()
                else None
            ),
            "near_elite_operator_eval_path": (
                str(
                    metrics_root
                    / f"{tag}_eval_near_elite_s128_seed3"
                    / "metrics.json"
                )
                if (
                    metrics_root
                    / f"{tag}_eval_near_elite_s128_seed3"
                    / "metrics.json"
                ).exists()
                else None
            ),
            "closed_loop_eval_path": None,
            "identity_check_passed": identity_ok,
            "teacher_anchor_source": teacher_anchor_source,
            "status": "ok" if random_metrics is not None else "partial",
            "notes": ";".join(notes),
        }
        if identity_ok is False:
            row["status"] = "INVALID_FRONTIER"
            row["notes"] = (
                (row["notes"] + ";" if row["notes"] else "")
                + "INVALID_FRONTIER: no-compression identity check failed."
            )
        rows.append(row)

    rows.sort(
        key=lambda r: (
            str(r.get("method_category")),
            -float(r["predictor_compression_ratio"])
            if r.get("predictor_compression_ratio") is not None
            else 1e9,
            str(r.get("tag")),
        )
    )

    csv_path = out_root / "rank_frontier_tworoom.csv"
    json_path = out_root / "rank_frontier_tworoom.json"
    md_path = out_root / "rank_frontier_tworoom.md"

    fields = [
        "tag",
        "method_category",
        "rank_fraction",
        "predictor_compression_ratio",
        "total_compression_ratio",
        "operator_spearman_random",
        "operator_top5_random",
        "operator_top10_random",
        "operator_regret_random",
        "first_action_error_random",
        "raw_cost_mse_random",
        "closed_loop_success_rate",
        "prediction_rollout_error",
        "identity_check_passed",
        "teacher_anchor_source",
        "status",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    payload = {
        "env": args.env,
        "source_of_truth": {
            "random_operator_eval": (
                "scripts/evaluate_operator_metrics.py "
                "via shared operator_eval.py"
            ),
            "dataset_action_operator_eval": (
                "scripts/evaluate_operator_metrics.py on dataset-action cache"
            ),
            "near_elite_operator_eval": (
                "scripts/evaluate_operator_metrics.py on near-elite cache"
            ),
            "closed_loop_eval": "scripts/benchmark_cost_model.py",
        },
        "identity_check_passed": identity_ok,
        "teacher_anchor_source": teacher_anchor_source,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_cols = fields
    lines = [
        "# Rank Frontier Summary (TwoRoom)",
        "",
        "|" + "|".join(md_cols) + "|",
        "|" + "|".join(["---"] * len(md_cols)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row.get(c)) for c in md_cols) + "|")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
