from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from oawc.config import get_run_config
from oawc.compression.low_rank import (
    count_parameters,
    replace_linears_with_svd,
)
from oawc.models.lewm_loader import load_lewm_from_hf
from oawc.paths import OUTPUT_ROOT, save_json


def phase2_dir(run_name: str, method_name: str) -> Path:
    path = OUTPUT_ROOT / "phase2" / run_name / method_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="pusht_weak")
    parser.add_argument("--rank-fraction", type=float, default=0.5)
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = get_run_config(args.run)

    method_name = f"svd_predictor_rank{int(args.rank_fraction * 100):03d}"
    out_dir = phase2_dir(cfg.run_name, method_name)

    model = load_lewm_from_hf(cfg.checkpoint_repo, device=args.device)
    model.eval()

    original_total_params = count_parameters(model)
    original_predictor_params = count_parameters(model.predictor)

    start = time.time()
    compressed_predictor, report = replace_linears_with_svd(
        model.predictor,
        rank_fraction=args.rank_fraction,
        min_rank=args.min_rank,
    )
    elapsed = time.time() - start

    model.predictor = compressed_predictor
    model.eval()

    compressed_total_params = count_parameters(model)
    compressed_predictor_params = count_parameters(model.predictor)

    model_path = out_dir / "model.pt"
    metadata_path = out_dir / "metadata.json"

    torch.save(model, model_path)

    metadata = {
        "run_name": cfg.run_name,
        "checkpoint_repo": cfg.checkpoint_repo,
        "method": "svd_low_rank_dense",
        "target": "predictor",
        "rank_fraction": args.rank_fraction,
        "min_rank": args.min_rank,
        "model_path": model_path,
        "original_total_params": original_total_params,
        "compressed_total_params": compressed_total_params,
        "total_param_ratio": compressed_total_params / original_total_params,
        "total_param_reduction": 1.0 - compressed_total_params / original_total_params,
        "original_predictor_params": original_predictor_params,
        "compressed_predictor_params": compressed_predictor_params,
        "predictor_param_ratio": compressed_predictor_params / original_predictor_params,
        "predictor_param_reduction": 1.0 - compressed_predictor_params / original_predictor_params,
        "compression_elapsed_sec": elapsed,
        "low_rank_report": report.to_dict(),
        "hardware_note": (
            "Dense low-rank factorization uses standard PyTorch Linear layers, "
            "so it is CPU/GPU compatible. Wall-clock speedup is not guaranteed "
            "because each original Linear becomes two dense GEMMs."
        ),
    }

    save_json(metadata_path, metadata)

    print("Saved compressed LeWM predictor:")
    print(f"  method: {method_name}")
    print(f"  model:  {model_path}")
    print(f"  meta:   {metadata_path}")
    print()
    print("Parameter summary:")
    print(f"  total:     {original_total_params:,} -> {compressed_total_params:,}")
    print(f"  predictor: {original_predictor_params:,} -> {compressed_predictor_params:,}")
    print(f"  predictor reduction: {100 * metadata['predictor_param_reduction']:.1f}%")
    print(f"  total reduction:     {100 * metadata['total_param_reduction']:.1f}%")
    print(f"  replaced linears:    {report.num_linear_replaced}")


if __name__ == "__main__":
    main()