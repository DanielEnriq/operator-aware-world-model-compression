from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from oawc.compression.operator_eval import evaluate_model_on_operator_cache
from oawc.compression.operator_metrics import resolve_device
from oawc.compression.reports import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument(
        "--dataset-action-cache",
        required=True,
        help="Operator cache built from dataset action windows.",
    )
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--student-model-path", required=True)
    parser.add_argument("--teacher-tag", default="teacher")
    parser.add_argument("--student-tag", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-candidates", type=int, default=128)
    args = parser.parse_args()

    device = resolve_device(args.device)
    teacher = evaluate_model_on_operator_cache(
        cache_path=args.dataset_action_cache,
        model_path=args.teacher_model_path,
        device=device,
        use_chunked_student=True,
        batch_states=int(args.batch_states),
        batch_candidates=int(args.batch_candidates),
    )
    student = evaluate_model_on_operator_cache(
        cache_path=args.dataset_action_cache,
        model_path=args.student_model_path,
        device=device,
        use_chunked_student=True,
        batch_states=int(args.batch_states),
        batch_candidates=int(args.batch_candidates),
    )
    teacher_costs = teacher["student_costs"]
    student_costs = student["student_costs"]
    mse = float(torch.mean((teacher_costs - student_costs) ** 2).item())
    mae = float(torch.mean((teacher_costs - student_costs).abs()).item())
    max_abs = float((teacher_costs - student_costs).abs().max().item())

    out_dir = Path("outputs/prediction_rollout") / args.env / args.student_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "env": args.env,
        "cache_path": args.dataset_action_cache,
        "teacher_tag": args.teacher_tag,
        "student_tag": args.student_tag,
        "teacher_model_path": args.teacher_model_path,
        "student_model_path": args.student_model_path,
        "device": device,
        "teacher_student_cost_mse": mse,
        "teacher_student_cost_mae": mae,
        "teacher_student_cost_max_abs": max_abs,
        "teacher_metadata": teacher["metadata"],
        "student_metadata": student["metadata"],
        "source_of_truth": (
            "dataset_action_operator_cache + shared operator_eval.py "
            "(cost-path fidelity proxy on dataset action windows)"
        ),
    }
    save_json(out_path, payload)
    print(f"[done] wrote: {out_path}")


if __name__ == "__main__":
    main()
