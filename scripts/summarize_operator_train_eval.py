from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from oawc.compression.reports import save_json


DEFAULT_MODEL_TAGS = [
    "lewm_tworoom_svd_r050",
    "lewm_tworoom_aa_svd_r050",
    "lewm_tworoom_svd_r050_cost_kl_split",
    "lewm_tworoom_svd_r050_elite_k10_split",
    "lewm_tworoom_svd_r050_hybrid_split",
    "lewm_tworoom_aa_svd_r025",
    "lewm_tworoom_aa_svd_r025_hybrid_split",
    "lewm_tworoom_svd_r025",
    "lewm_tworoom_svd_r025_hybrid_split",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_mean(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        mean = value.get("mean")
        return float(mean) if mean is not None else None
    if value is None:
        return None
    return float(value)


def _topk_mean(metrics: dict[str, Any], k: str) -> float | None:
    item = metrics.get("topk_overlap", {}).get(k, {})
    mean = item.get("mean")
    return float(mean) if mean is not None else None


def _method_and_rank(
    base_tag: str,
    compression_root: Path,
) -> tuple[str | None, float | None]:
    comp_path = compression_root / base_tag / "compression_report.json"
    distill_path = compression_root / base_tag / "distillation_report.json"
    method = None
    rank_fraction = None
    if comp_path.exists():
        comp = _read_json(comp_path)
        method = comp.get("method")
        if comp.get("rank_fraction") is not None:
            rank_fraction = float(comp["rank_fraction"])
    if distill_path.exists():
        dist = _read_json(distill_path)
        method = dist.get("method") or method
        inherited = dist.get("inherited_compression", {})
        if inherited.get("rank_fraction") is not None:
            rank_fraction = float(inherited["rank_fraction"])
    return method, rank_fraction


def _generalization_note(
    train_s: float | None,
    eval_s: float | None,
    train_t5: float | None,
    eval_t5: float | None,
    train_r: float | None,
    eval_r: float | None,
) -> str:
    notes: list[str] = []
    if train_s is None or eval_s is None:
        notes.append("missing_spearman")
    else:
        ds = eval_s - train_s
        if ds < -0.20:
            notes.append("large_spearman_drop")
        elif ds < -0.10:
            notes.append("moderate_spearman_drop")
        else:
            notes.append("spearman_stableish")
    if train_t5 is not None and eval_t5 is not None:
        dt = eval_t5 - train_t5
        if dt < -0.10:
            notes.append("large_top5_drop")
        elif dt < -0.05:
            notes.append("moderate_top5_drop")
    if train_r is not None and eval_r is not None:
        dr = eval_r - train_r
        if dr > 20.0:
            notes.append("large_regret_increase")
        elif dr > 5.0:
            notes.append("moderate_regret_increase")
    return ";".join(notes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument(
        "--metrics-root",
        default="outputs/operator_metrics/tworoom",
    )
    parser.add_argument(
        "--compression-root",
        default="outputs/compression/tworoom",
    )
    parser.add_argument(
        "--train-suffix",
        default="train_s512_seed0",
    )
    parser.add_argument(
        "--eval-suffix",
        default="eval_s128_seed1",
    )
    parser.add_argument(
        "--model-tags",
        nargs="*",
        default=DEFAULT_MODEL_TAGS,
    )
    args = parser.parse_args()

    metrics_root = Path(args.metrics_root)
    compression_root = Path(args.compression_root)
    if not metrics_root.exists():
        raise FileNotFoundError(f"Missing metrics root: {metrics_root}")

    rows: list[dict[str, Any]] = []
    for base_tag in args.model_tags:
        train_tag = f"{base_tag}_{args.train_suffix}"
        eval_tag = f"{base_tag}_{args.eval_suffix}"
        train_metrics_path = metrics_root / train_tag / "metrics.json"
        eval_metrics_path = metrics_root / eval_tag / "metrics.json"
        train_metrics = (
            _read_json(train_metrics_path) if train_metrics_path.exists() else {}
        )
        eval_metrics = (
            _read_json(eval_metrics_path) if eval_metrics_path.exists() else {}
        )
        method, rank_fraction = _method_and_rank(base_tag, compression_root)

        train_s = (
            _metric_mean(train_metrics, "spearman_per_state")
            if train_metrics
            else None
        )
        eval_s = (
            _metric_mean(eval_metrics, "spearman_per_state")
            if eval_metrics
            else None
        )
        train_t5 = _topk_mean(train_metrics, "5") if train_metrics else None
        eval_t5 = _topk_mean(eval_metrics, "5") if eval_metrics else None
        train_r = (
            _metric_mean(train_metrics, "teacher_regret")
            if train_metrics
            else None
        )
        eval_r = (
            _metric_mean(eval_metrics, "teacher_regret")
            if eval_metrics
            else None
        )

        row = {
            "base_tag": base_tag,
            "method": method,
            "rank_fraction": rank_fraction,
            "train_tag": train_tag,
            "eval_tag": eval_tag,
            "train_spearman": train_s,
            "eval_spearman": eval_s,
            "train_top5": train_t5,
            "eval_top5": eval_t5,
            "train_regret": train_r,
            "eval_regret": eval_r,
            "generalization_gap_notes": _generalization_note(
                train_s,
                eval_s,
                train_t5,
                eval_t5,
                train_r,
                eval_r,
            ),
        }
        rows.append(row)

    rows.sort(key=lambda r: (str(r.get("method")), str(r.get("base_tag"))))
    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "operator_train_eval_summary_tworoom.csv"
    json_path = out_dir / "operator_train_eval_summary_tworoom.json"
    md_path = out_dir / "operator_train_eval_summary_tworoom.md"

    fields = [
        "base_tag",
        "method",
        "rank_fraction",
        "train_spearman",
        "eval_spearman",
        "train_top5",
        "eval_top5",
        "train_regret",
        "eval_regret",
        "generalization_gap_notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    save_json(
        json_path,
        {
            "env": args.env,
            "train_suffix": args.train_suffix,
            "eval_suffix": args.eval_suffix,
            "rows": rows,
        },
    )

    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "# Operator Train vs Eval Summary (TwoRoom)",
        "",
        "|" + "|".join(fields) + "|",
        "|" + "|".join(["---"] * len(fields)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(fmt(row.get(k)) for k in fields) + "|")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Operator train/eval summary written")
    print(f"  csv:  {csv_path}")
    print(f"  json: {json_path}")
    print(f"  md:   {md_path}")


if __name__ == "__main__":
    main()
