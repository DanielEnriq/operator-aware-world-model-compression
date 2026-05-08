from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

METHOD_LABELS = {
    "teacher": "Teacher",
    "weight_svd": "Weight SVD",
    "activation_svd": "Activation-aware SVD",
    "operator_cost_kl": "Operator-aware KL",
    "operator_hybrid": "Operator-aware Hybrid",
    "operator_elite": "Operator-aware Elite",
}
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


def _enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    bench_path = out.get("benchmark_json")
    if isinstance(bench_path, str) and Path(bench_path).exists():
        bench = _read_json(Path(bench_path))
        perf = bench.get("performance", {})
        eff = bench.get("efficiency", {})
        if out.get("success_rate") is None:
            out["success_rate"] = perf.get("success_rate")
        if out.get("num_successes") is None:
            out["num_successes"] = perf.get("num_successes")
        if out.get("avg_return") is None:
            out["avg_return"] = perf.get("avg_return")
        if out.get("avg_final_distance") is None:
            out["avg_final_distance"] = perf.get("avg_final_distance")
        if out.get("eval_time_s") is None:
            out["eval_time_s"] = eff.get("evaluation_time_sec")

    comp_report = out.get("compression_report")
    if isinstance(comp_report, str) and Path(comp_report).exists():
        comp = _read_json(Path(comp_report))
        out["predictor_compression_ratio"] = out.get(
            "predictor_compression_ratio",
            comp.get("predictor_compression_ratio"),
        )
        out["total_compression_ratio"] = out.get(
            "total_compression_ratio",
            comp.get("total_compression_ratio"),
        )
        out["compression_status"] = out.get(
            "compression_status",
            comp.get("compression_status"),
        )

    distill_report = out.get("distill_report")
    if isinstance(distill_report, str) and Path(distill_report).exists():
        distill = _read_json(Path(distill_report))
        if out.get("distill_steps") is None:
            out["distill_steps"] = distill.get("distill_steps", distill.get("max_steps"))
        if out.get("distill_batch_size") is None:
            out["distill_batch_size"] = distill.get("batch_size")
        if out.get("distill_train_cache") is None:
            out["distill_train_cache"] = distill.get("train_cache")
        if out.get("distill_loss") is None:
            out["distill_loss"] = distill.get(
                "final_training_loss",
                distill.get("final_total_loss", distill.get("final_train_kl")),
            )
        if out.get("distill_wall_time_s") is None:
            out["distill_wall_time_s"] = distill.get("wall_time_sec")
    return out


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("method")),
            -float(row["rank_fraction"])
            if isinstance(row.get("rank_fraction"), (int, float))
            else 0.0,
        ),
    )


def _best_per_band(
    rows: list[dict[str, Any]],
    *,
    ratio_key: str,
    bands: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and isinstance(row.get("success_rate"), (int, float))
        and isinstance(row.get(ratio_key), (int, float))
    ]
    for low, high in bands:
        candidates = [
            row
            for row in ok_rows
            if low <= float(row[ratio_key]) < high
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: float(row["success_rate"]))
        winners.append(
            {
                "band": f"[{low:.2f},{high:.2f})",
                "tag_method": f"{best.get('method')}@r{best.get('rank_fraction')}",
                "success_rate": best.get("success_rate"),
                ratio_key: best.get(ratio_key),
            }
        )
    return winners


def _within_95pct_teacher(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    teacher_rows = [
        row
        for row in rows
        if row.get("method") == "teacher"
        and isinstance(row.get("success_rate"), (int, float))
    ]
    if not teacher_rows:
        return None
    teacher_sr = float(teacher_rows[0]["success_rate"])
    threshold = 0.95 * teacher_sr
    candidates = [
        row
        for row in rows
        if row.get("method") != "teacher"
        and row.get("status") == "ok"
        and isinstance(row.get("success_rate"), (int, float))
        and float(row["success_rate"]) >= threshold
        and isinstance(row.get("total_compression_ratio"), (int, float))
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row["total_compression_ratio"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        required=True,
        help="Runner output JSON containing canonical rows.",
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--num-eval", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="outputs/tables")
    args = parser.parse_args()

    payload = _read_json(Path(args.input_json))
    rows_raw = list(payload.get("rows", []))
    rows = [_enrich_row(row) for row in rows_raw]
    rows = [
        row
        for row in rows
        if row.get("env") == args.env and row.get("model_family") == args.model_family
    ]
    rows = _sort_rows(rows)

    out_base = (
        Path(args.output_root)
        / f"full_closed_loop_frontier_{args.env}_{args.model_family}"
    )
    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_base.with_suffix(".csv")
    json_path = out_base.with_suffix(".json")
    md_path = out_base.with_suffix(".md")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in TABLE_COLUMNS})

    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "full_closed_loop_frontier_summary",
                "env": args.env,
                "model_family": args.model_family,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    failures = [row for row in rows if row.get("status") not in {"ok", "skipped_existing"}]
    band_winners = _best_per_band(
        rows,
        ratio_key="total_compression_ratio",
        bands=[(0.0, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.01)],
    )
    within_95 = _within_95pct_teacher(rows)
    lines = [
        f"# Full Closed-Loop Frontier Summary ({args.env}, {args.model_family})",
        "",
        f"n={args.num_eval}, seed={args.seed}.",
        "Closed-loop MPC success is the primary metric; operator cache is training-only signal.",
        "Preliminary note: n=64 is preliminary unless larger n is used.",
        "",
        "## Main table",
        "|" + "|".join(TABLE_COLUMNS) + "|",
        "|" + "|".join(["---"] * len(TABLE_COLUMNS)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_fmt(row.get(c)) for c in TABLE_COLUMNS) + "|")

    lines.append("")
    lines.append("## Best by compression band")
    if band_winners:
        lines.append("|band|winner|success_rate|total_compression_ratio|")
        lines.append("|---|---|---|---|")
        for winner in band_winners:
            lines.append(
                f"|{winner['band']}|{winner['tag_method']}|"
                f"{_fmt(winner['success_rate'])}|{_fmt(winner['total_compression_ratio'])}|"
            )
    else:
        lines.append("No successful compressed rows found.")

    lines.append("")
    lines.append("## Most compressed within 95% teacher success")
    if within_95 is not None:
        lines.append(
            "- "
            f"{METHOD_LABELS.get(str(within_95.get('method')), str(within_95.get('method')))} "
            f"(rank={within_95.get('rank_fraction')}, "
            f"success_rate={_fmt(within_95.get('success_rate'))}, "
            f"total_ratio={_fmt(within_95.get('total_compression_ratio'))})"
        )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("## Failure rows")
    if failures:
        lines.append("|method|rank_fraction|status|error_message|")
        lines.append("|---|---|---|---|")
        for row in failures:
            lines.append(
                f"|{_fmt(row.get('method'))}|{_fmt(row.get('rank_fraction'))}|"
                f"{_fmt(row.get('status'))}|{_fmt(row.get('error_message'))}|"
            )
    else:
        lines.append("None.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
