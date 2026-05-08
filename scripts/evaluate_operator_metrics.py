from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from oawc.compression.operator_eval import (
    evaluate_model_on_operator_cache,
)
from oawc.compression.operator_metrics import resolve_device
from oawc.compression.reports import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    env_name = cache["env"]

    device = resolve_device(args.device)
    topk_keys = sorted(int(k) for k in cache.get("topk_indices", {}).keys())
    result = evaluate_model_on_operator_cache(
        cache_path=cache_path,
        model_path=args.model_path,
        device=device,
        use_chunked_student=False,
        topk=topk_keys if topk_keys else None,
    )
    metrics_core = result["metrics"]

    output_dir = Path("outputs/operator_metrics") / env_name / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics.json"

    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "env": env_name,
        "tag": args.tag,
        "cache_path": str(cache_path),
        "model_path": str(Path(args.model_path)),
        "device": device,
        **metrics_core,
    }
    save_json(out_path, metrics)

    print("Operator metrics evaluation complete")
    print(f"  tag:                         {args.tag}")
    print(f"  finite student costs:        {metrics['finite_student_costs']}")
    print(f"  raw cost mse:                {metrics['raw_cost_mse']:.6f}")
    print(
        "  teacher best index match:    "
        f"{metrics['teacher_best_index_match_rate']:.6f}"
    )
    print(
        "  spearman mean:               "
        f"{metrics['spearman_per_state']['mean']:.6f}"
    )
    print(f"  saved:                       {out_path}")


if __name__ == "__main__":
    main()
