from __future__ import annotations

import argparse
from pathlib import Path

from oawc.benchmark import load_hdf5_dataset
from oawc.models import load_cost_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument(
        "--teacher-checkpoint",
        default="quentinll/lewm-tworooms",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--train-cache",
        default=(
            "outputs/operator_cache/tworoom/"
            "lewm_tworoom_train_s512_c128_seed0/operator_cache.pt"
        ),
    )
    parser.add_argument(
        "--eval-cache",
        default=(
            "outputs/operator_cache/tworoom/"
            "lewm_tworoom_eval_s128_c128_seed1/operator_cache.pt"
        ),
    )
    args = parser.parse_args()

    print("[check] loading dataset...")
    dataset = load_hdf5_dataset(args.env)
    print(
        "[ok] dataset loaded:",
        f"columns={list(dataset.column_names)}",
        f"rows={len(dataset)}",
    )

    print("[check] loading teacher model...")
    loaded = load_cost_model(
        family="lewm_hf",
        checkpoint=args.teacher_checkpoint,
        env_name=args.env,
        device=args.device,
    )
    print("[ok] teacher model loaded:", type(loaded.model).__name__)

    train_cache = Path(args.train_cache)
    eval_cache = Path(args.eval_cache)
    if train_cache.exists():
        print("[ok] train cache exists:", train_cache)
    else:
        print("[missing] train cache:", train_cache)
        print(
            "  build cmd:",
            "uv run python scripts/build_operator_cache.py "
            f"--env {args.env} --model-family lewm_hf "
            f"--checkpoint {args.teacher_checkpoint} --num-states 512 "
            "--num-candidates 128 --horizon 5 --topk 1 5 10 20 --seed 0 "
            "--tag lewm_tworoom_train_s512_c128_seed0",
        )

    if eval_cache.exists():
        print("[ok] eval cache exists:", eval_cache)
    else:
        print("[missing] eval cache:", eval_cache)
        print(
            "  build cmd:",
            "uv run python scripts/build_operator_cache.py "
            f"--env {args.env} --model-family lewm_hf "
            f"--checkpoint {args.teacher_checkpoint} --num-states 128 "
            "--num-candidates 128 --horizon 5 --topk 1 5 10 20 --seed 1 "
            "--tag lewm_tworoom_eval_s128_c128_seed1",
        )

    print("[check] rank-tag formatting dry-run")
    for rf in [1.0, 0.95, 0.90, 0.85, 0.5]:
        scaled = int(round(rf * 100))
        print(f"  rank_fraction={rf:.2f} -> r{scaled:03d}")

    print("[ok] smoke_rank_frontier_setup complete")


if __name__ == "__main__":
    main()
