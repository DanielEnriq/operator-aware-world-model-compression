from __future__ import annotations

import argparse
import json
from pathlib import Path

from oawc.config import get_run_config
from oawc.paths import OUTPUT_ROOT, phase1_dirs


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def fmt_millions(x: float | int) -> str:
    return f"{x / 1_000_000:.2f}M"


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def summarize_svd(run_name: str) -> None:
    cfg = get_run_config(run_name)

    phase1 = phase1_dirs(cfg)
    results_dir = phase1["results"]
    phase2_dir = OUTPUT_ROOT / "phase2" / cfg.run_name

    original_metrics_path = results_dir / "lewm_transition_metrics_original_norm_actions.json"

    if not original_metrics_path.exists():
        raise FileNotFoundError(
            f"Missing original metrics: {original_metrics_path}\n"
            "Run phase1_lewm_transition_metrics.py with --tag original first."
        )

    original_metrics = load_json(original_metrics_path)

    rows = []

    original_final_mse = original_metrics["final_transition_mse"]
    original_seq_mse = original_metrics["sequence_transition_mse"]
    copy_final_mse = original_metrics["copy_final_baseline_mse"]

    rows.append(
        {
            "method": "original",
            "rank_fraction": 1.0,
            "total_params": None,
            "predictor_params": None,
            "total_reduction": 0.0,
            "predictor_reduction": 0.0,
            "final_mse": original_final_mse,
            "seq_mse": original_seq_mse,
            "rel_final_mse": 1.0,
            "copy_gap": original_final_mse / copy_final_mse,
            "elapsed_sec": original_metrics.get("elapsed_sec"),
        }
    )

    if phase2_dir.exists():
        for method_dir in sorted(phase2_dir.glob("svd_predictor_rank*")):
            metadata_path = method_dir / "metadata.json"
            tag = method_dir.name
            metrics_path = results_dir / f"lewm_transition_metrics_{tag}_norm_actions.json"

            if not metadata_path.exists() or not metrics_path.exists():
                continue

            metadata = load_json(metadata_path)
            metrics = load_json(metrics_path)

            final_mse = metrics["final_transition_mse"]
            seq_mse = metrics["sequence_transition_mse"]

            rows.append(
                {
                    "method": tag,
                    "rank_fraction": metadata["rank_fraction"],
                    "total_params": metadata["compressed_total_params"],
                    "predictor_params": metadata["compressed_predictor_params"],
                    "total_reduction": metadata["total_param_reduction"],
                    "predictor_reduction": metadata["predictor_param_reduction"],
                    "final_mse": final_mse,
                    "seq_mse": seq_mse,
                    "rel_final_mse": final_mse / original_final_mse,
                    "copy_gap": final_mse / copy_final_mse,
                    "elapsed_sec": metrics.get("elapsed_sec"),
                }
            )

    rows = sorted(rows, key=lambda r: r["rank_fraction"], reverse=True)

    print()
    print(f"SVD predictor compression summary for run={cfg.run_name}")
    print(f"Original final transition MSE: {original_final_mse:.6f}")
    print(f"Copy-last final baseline MSE: {copy_final_mse:.6f}")
    print()

    header = (
        f"{'method':<26}"
        f"{'rank':>8}"
        f"{'total params':>14}"
        f"{'pred params':>14}"
        f"{'total red.':>12}"
        f"{'pred red.':>11}"
        f"{'final MSE':>12}"
        f"{'rel MSE':>10}"
        f"{'copy ratio':>11}"
        f"{'time(s)':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        total_params = "—" if r["total_params"] is None else fmt_millions(r["total_params"])
        pred_params = "—" if r["predictor_params"] is None else fmt_millions(r["predictor_params"])
        elapsed = "—" if r["elapsed_sec"] is None else f"{r['elapsed_sec']:.2f}"

        print(
            f"{r['method']:<26}"
            f"{r['rank_fraction']:>8.2f}"
            f"{total_params:>14}"
            f"{pred_params:>14}"
            f"{fmt_pct(r['total_reduction']):>12}"
            f"{fmt_pct(r['predictor_reduction']):>11}"
            f"{r['final_mse']:>12.6f}"
            f"{r['rel_final_mse']:>10.3f}"
            f"{r['copy_gap']:>11.3f}"
            f"{elapsed:>9}"
        )

    print()
    print("Interpretation:")
    print("  rel MSE = compressed final_transition_mse / original final_transition_mse.")
    print("  copy ratio = model final_transition_mse / copy-last baseline MSE.")
    print("  copy ratio < 1 means the model still beats the copy-last baseline.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    args = parser.parse_args()

    summarize_svd(args.run)


if __name__ == "__main__":
    main()
