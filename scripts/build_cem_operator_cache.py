from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument("--model-family", default="lewm_hf")
    parser.add_argument("--checkpoint", default="quentinll/lewm-tworooms")
    parser.add_argument("--num-states", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--cem-iters", type=int, default=3)
    parser.add_argument("--cem-pop-size", type=int, default=128)
    parser.add_argument("--cem-elite-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--tag", required=True)
    parser.parse_args()
    raise NotImplementedError(
        "build_cem_operator_cache.py is intentionally scaffolded. "
        "For the next iteration, use build_near_elite_operator_cache.py "
        "to approximate planner-induced candidates with elite perturbations."
    )


if __name__ == "__main__":
    main()
