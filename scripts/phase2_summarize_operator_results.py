from __future__ import annotations

import argparse
import json
from pathlib import Path

from oawc.config import get_run_config
from oawc.paths import OUTPUT_ROOT, phase1_dirs


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def fmt_millions(x: int | float | None) -> str:
    if x is None:
        return "—"
    return f"{x / 1_000_000:.2f}M"


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.1f}%"


def summarize(run_name: str) -> None:
    cfg = get_run_config(run_name)

    phase1 = phase1_dirs(cfg)
    results_dir = phase1["results"]
    phase2_dir = OUTPUT_ROOT / "phase2" / cfg.run_name

    original_path = results_dir / "lewm_transition_metrics_original_norm_actions.json"
    if not original_path.exists():
        raise FileNotFoundError(f"Missing original transition metrics: {original_path}")

    original = load_json(original_path)
    original_final_mse = original["final_transition_mse"]
    copy_final_mse = original["copy_final_baseline_mse"]

    rows = [
        {
            "method": "original",
            "rank": 1.0,
            "total_params": None,
            "pred_params": None,
            "total_red": 0.0,
            "pred_red": 0.0,
            "final_mse": original_final_mse,
            "rel_mse": 1.0,
            "copy_ratio": original_final_mse / copy_final_mse,
            "spearman": None,
            "top1": None,
            "topk": None,
            "latency_ratio": 1.0,
            "cand_per_sec": None,
        }
    ]

    def add_method_row(method_dir: Path, phase: str) -> None:
        tag = method_dir.name

        meta_path = method_dir / "metadata.json"
        transition_path = results_dir / f"lewm_transition_metrics_{tag}_norm_actions.json"
        ranking_path = method_dir / "ranking_eval.json"

        if not meta_path.exists() or not transition_path.exists():
            return

        meta = load_json(meta_path)
        transition = load_json(transition_path)
        ranking = load_json(ranking_path) if ranking_path.exists() else {}

        final_mse = transition["final_transition_mse"]

        # Phase 2 metadata has direct compression counts.
        # Phase 3 metadata points back to a base compressed model.
        if phase == "phase2":
            rank = meta.get("rank_fraction")
            total_params = meta.get("compressed_total_params")
            pred_params = meta.get("compressed_predictor_params")
            total_red = meta.get("total_param_reduction")
            pred_red = meta.get("predictor_param_reduction")
        else:
            base_path = Path(meta["base_compressed_model_path"])
            base_meta_path = base_path.parent / "metadata.json"
            base_meta = load_json(base_meta_path)

            rank = base_meta.get("rank_fraction")
            total_params = base_meta.get("compressed_total_params")
            pred_params = base_meta.get("compressed_predictor_params")
            total_red = base_meta.get("total_param_reduction")
            pred_red = base_meta.get("predictor_param_reduction")

        rows.append(
            {
                "method": tag,
                "rank": rank,
                "total_params": total_params,
                "pred_params": pred_params,
                "total_red": total_red,
                "pred_red": pred_red,
                "final_mse": final_mse,
                "rel_mse": final_mse / original_final_mse,
                "copy_ratio": final_mse / copy_final_mse,
                "spearman": ranking.get("spearman_mean"),
                "top1": ranking.get("top1_agreement"),
                "topk": ranking.get("topk_overlap"),
                "latency_ratio": ranking.get("latency_ratio_compressed_over_original"),
                "cand_per_sec": ranking.get("compressed_candidates_per_sec"),
            }
        )


    for method_dir in sorted(phase2_dir.glob("svd_predictor_rank*")):
        add_method_row(method_dir, phase="phase2")

    phase3_dir = OUTPUT_ROOT / "phase3" / cfg.run_name
    if phase3_dir.exists():
        for method_dir in sorted(phase3_dir.glob("*")):
            add_method_row(method_dir, phase="phase3")
        rows = sorted(rows, key=lambda r: r["rank"], reverse=True)

    print()
    print(f"Operator-aware compression summary for run={cfg.run_name}")
    print(f"Original final transition MSE: {original_final_mse:.6f}")
    print(f"Copy-last final baseline MSE: {copy_final_mse:.6f}")
    print()

    header = (
        f"{'method':<26}"
        f"{'rank':>7}"
        f"{'total':>10}"
        f"{'pred':>10}"
        f"{'tot red':>9}"
        f"{'pred red':>10}"
        f"{'final MSE':>11}"
        f"{'rel MSE':>9}"
        f"{'rho':>8}"
        f"{'top1':>8}"
        f"{'top5':>8}"
        f"{'lat x':>8}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        rho = "—" if r["spearman"] is None else f"{r['spearman']:.3f}"
        top1 = "—" if r["top1"] is None else f"{r['top1']:.3f}"
        topk = "—" if r["topk"] is None else f"{r['topk']:.3f}"
        lat = "—" if r["latency_ratio"] is None else f"{r['latency_ratio']:.3f}"

        print(
            f"{r['method']:<26}"
            f"{r['rank']:>7.2f}"
            f"{fmt_millions(r['total_params']):>10}"
            f"{fmt_millions(r['pred_params']):>10}"
            f"{fmt_pct(r['total_red']):>9}"
            f"{fmt_pct(r['pred_red']):>10}"
            f"{r['final_mse']:>11.6f}"
            f"{r['rel_mse']:>9.3f}"
            f"{rho:>8}"
            f"{top1:>8}"
            f"{topk:>8}"
            f"{lat:>8}"
        )

    print()
    print("Notes:")
    print("  final MSE is teacher-forced transition error against future LeWM embeddings.")
    print("  rho is Spearman rank correlation between original and compressed planning costs.")
    print("  top1/top5 measure preservation of the best / top-5 candidate action choices.")
    print("  lat x is compressed latency divided by original latency; <1 means faster.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    args = parser.parse_args()

    summarize(args.run)


if __name__ == "__main__":
    main()
