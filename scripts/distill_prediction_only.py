from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from oawc.benchmark import load_hdf5_dataset
from oawc.compression.operator_metrics import (
    adapt_candidates_for_model,
    build_info_dict_from_cache,
    compute_model_costs,
    expand_info_for_candidates,
    load_model_from_path,
    maybe_align_action_width,
    resolve_device,
)
from oawc.compression.prediction_distill import (
    build_prediction_batch,
    clone_info_dict,
    load_inherited_compression_report,
    predictor_param_partition,
    sample_valid_row_indices,
    set_trainable_by_substring,
)
from oawc.compression.reports import save_json
from oawc.models import load_cost_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--teacher-family", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--dataset-source", default="offline")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--trainable-substring", default="predictor")
    parser.add_argument(
        "--save-best-by",
        default="finite_operator_check",
        choices=["finite_operator_check"],
    )
    parser.add_argument("--skip-nonfinite-batches", action="store_true")
    parser.add_argument("--no-skip-nonfinite-batches", dest="skip_nonfinite_batches", action="store_false")
    parser.set_defaults(skip_nonfinite_batches=True)
    parser.add_argument("--restore-best-on-failure", action="store_true")
    parser.add_argument("--no-restore-best-on-failure", dest="restore_best_on_failure", action="store_false")
    parser.set_defaults(restore_best_on_failure=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-root", default="outputs/compression")
    return parser.parse_args()


def _summary_stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "max": float(x.max().item()),
    }


def _load_baseline_metrics(student_path: str | Path) -> tuple[str | None, dict | None]:
    student_path = Path(student_path)
    baseline_tag = student_path.parent.name
    env_name = student_path.parent.parent.name
    metrics_path = (
        Path("outputs/operator_metrics")
        / env_name
        / baseline_tag
        / "metrics.json"
    )
    if not metrics_path.exists():
        return None, None
    return str(metrics_path), json.loads(metrics_path.read_text())


def _build_small_operator_batch(
    *,
    env: str,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, str]:
    smoke_path = Path(
        "outputs/operator_cache/tworoom/lewm_tworoom_smoke/operator_cache.pt"
    )
    medium_path = Path(
        "outputs/operator_cache/tworoom/lewm_tworoom_medium/operator_cache.pt"
    )
    cache_path = smoke_path if smoke_path.exists() else medium_path
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    n_states = min(4, int(cache["num_states"]))
    n_candidates = min(16, int(cache["num_candidates"]))
    episodes_idx = list(cache["episodes_idx"][:n_states])
    start_steps = list(cache["start_steps"][:n_states])
    goal_offset_steps = int(cache["goal_offset_steps"])
    info = build_info_dict_from_cache(
        env_name=env,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset_steps=goal_offset_steps,
        device=device,
    )
    candidate_actions = (
        cache["candidate_actions"][:n_states, :n_candidates]
        .float()
        .to(device)
    )
    return info, candidate_actions, str(cache_path)


def _operator_sanity_check(
    *,
    model: torch.nn.Module,
    env: str,
    device: str,
) -> dict:
    info, candidate_actions, cache_path = _build_small_operator_batch(
        env=env,
        device=device,
    )
    info = maybe_align_action_width(info, model)
    candidate_eval = adapt_candidates_for_model(candidate_actions, model)
    expanded = expand_info_for_candidates(
        info,
        num_candidates=int(candidate_eval.shape[1]),
    )
    with torch.no_grad():
        costs = compute_model_costs(model, expanded, candidate_eval)
    costs = costs.detach().cpu().float()
    finite = bool(torch.isfinite(costs).all().item())
    std = float(costs.std().item()) if finite else float("nan")
    if finite:
        best_idx = torch.argmin(costs, dim=1)
        unique_best = int(torch.unique(best_idx).numel())
    else:
        best_idx = torch.zeros(costs.shape[0], dtype=torch.long)
        unique_best = 0
    constant_best = bool(unique_best == 1 and costs.shape[0] > 1)
    passed = bool(finite and std > 1e-6 and not constant_best)
    return {
        "passed": passed,
        "finite_student_costs": finite,
        "student_cost_std": std,
        "student_best_constant": constant_best,
        "unique_student_best_index_count": unique_best,
        "student_best_index": best_idx.tolist(),
        "cache_used_for_validation": cache_path,
    }


def main() -> None:
    args = parse_args()
    if args.dataset_source != "offline":
        raise ValueError(
            "Method 3 v0 currently supports --dataset-source offline"
        )

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_root) / args.env / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    teacher_loaded = load_cost_model(
        family=args.teacher_family,
        checkpoint=args.teacher_checkpoint,
        env_name=args.env,
        device=device,
    )
    teacher = teacher_loaded.model.to(device).eval()
    teacher.requires_grad_(False)

    student = load_model_from_path(args.student_path, device=device)
    student.train()
    trainable_params = set_trainable_by_substring(
        student,
        args.trainable_substring,
    )
    if not trainable_params:
        raise ValueError(
            "No trainable parameters matched "
            f"--trainable-substring={args.trainable_substring!r}"
        )
    num_trainable, num_frozen = predictor_param_partition(
        student,
        args.trainable_substring,
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def snapshot_trainable_state() -> dict[str, torch.Tensor]:
        return {
            name: p.detach().cpu().clone()
            for name, p in student.named_parameters()
            if p.requires_grad
        }

    def restore_trainable_state(state: dict[str, torch.Tensor]) -> None:
        named = dict(student.named_parameters())
        for name, saved in state.items():
            named[name].data.copy_(saved.to(device=named[name].device))

    def has_nonfinite_trainable_params() -> bool:
        for p in trainable_params:
            if not torch.isfinite(p).all():
                return True
        return False

    last_good_state = snapshot_trainable_state()

    dataset = load_hdf5_dataset(args.env)

    first_step_loss_requires_grad = False
    first_step_nonzero_grad_params = 0
    first_step_total_grad_norm = 0.0

    initial_loss = None
    final_loss = None
    first_pred_shape = None
    skipped_nonfinite_batches = 0
    nonfinite_operator_eval_count = 0
    best_valid_loss = float("inf")
    best_valid_state = None
    best_valid_operator = None
    max_steps = int(args.max_steps)
    if args.max_epochs is not None and args.limit_batches is not None:
        max_steps = min(max_steps, int(args.max_epochs * args.limit_batches))

    with log_path.open("w", encoding="utf-8") as log_f:
        for step in range(1, max_steps + 1):
            row_indices = sample_valid_row_indices(
                dataset,
                horizon=args.horizon,
                batch_size=args.batch_size,
                rng=np_rng,
            )
            info, action_seq = build_prediction_batch(
                dataset,
                row_indices=row_indices,
                horizon=args.horizon,
                device=device,
            )
            teacher_action = adapt_candidates_for_model(action_seq, teacher)
            student_action = adapt_candidates_for_model(action_seq, student)

            with torch.no_grad():
                t_out = teacher.rollout(clone_info_dict(info), teacher_action)
                teacher_pred = t_out["predicted_emb"].detach()

            s_out = student.rollout(clone_info_dict(info), student_action)
            student_pred = s_out["predicted_emb"]
            if first_pred_shape is None:
                first_pred_shape = list(student_pred.shape)

            loss = F.mse_loss(student_pred, teacher_pred)
            if not torch.isfinite(loss):
                skipped_nonfinite_batches += 1
                teacher_is_finite = bool(torch.isfinite(teacher_pred).all().item())
                student_is_finite = bool(torch.isfinite(student_pred).all().item())
                restore_trainable_state(last_good_state)
                for group in optimizer.param_groups:
                    group["lr"] = float(group["lr"]) * 0.5
                rec = {
                    "step": step,
                    "loss": float("nan"),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "grad_norm": float("nan"),
                    "skipped_nonfinite_loss": True,
                    "reverted_to_last_good_state": True,
                    "teacher_pred_finite": teacher_is_finite,
                    "student_pred_finite": student_is_finite,
                    "teacher_pred_stats": (
                        _summary_stats(teacher_pred) if teacher_is_finite else None
                    ),
                    "student_pred_stats": (
                        _summary_stats(student_pred) if student_is_finite else None
                    ),
                    "row_indices": row_indices.tolist(),
                }
                log_f.write(json.dumps(rec) + "\n")
                if step % max(1, args.log_every) == 0 or step == 1:
                    print(
                        f"[train] step={step} skipped: non-finite loss, "
                        f"new_lr={optimizer.param_groups[0]['lr']:.3e}"
                    )
                if args.skip_nonfinite_batches:
                    continue
                raise RuntimeError(f"Non-finite loss at step {step}.")
            if initial_loss is None:
                initial_loss = float(loss.item())
            final_loss = float(loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            total_grad_sq = 0.0
            nonzero_grad = 0
            for p in trainable_params:
                if p.grad is None:
                    continue
                gn = float(p.grad.norm().item())
                if gn > 0:
                    nonzero_grad += 1
                total_grad_sq += gn * gn
            grad_norm = float(total_grad_sq ** 0.5)
            if not np.isfinite(grad_norm):
                skipped_nonfinite_batches += 1
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "grad_norm": float("nan"),
                    "skipped_nonfinite_grad": True,
                }
                optimizer.zero_grad(set_to_none=True)
                log_f.write(json.dumps(rec) + "\n")
                if step % max(1, args.log_every) == 0 or step == 1:
                    print(f"[train] step={step} skipped: non-finite grad")
                if args.skip_nonfinite_batches:
                    continue
                raise RuntimeError(f"Non-finite grad at step {step}.")

            if step == 1:
                first_step_loss_requires_grad = bool(loss.requires_grad)
                first_step_nonzero_grad_params = int(nonzero_grad)
                first_step_total_grad_norm = grad_norm

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    args.grad_clip,
                )
            optimizer.step()

            if has_nonfinite_trainable_params():
                skipped_nonfinite_batches += 1
                restore_trainable_state(last_good_state)
                for group in optimizer.param_groups:
                    group["lr"] = float(group["lr"]) * 0.5
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "grad_norm": grad_norm,
                    "reverted_nonfinite_params": True,
                }
                log_f.write(json.dumps(rec) + "\n")
                if step % max(1, args.log_every) == 0 or step == 1:
                    print(
                        f"[train] step={step} reverted: non-finite params, "
                        f"new_lr={optimizer.param_groups[0]['lr']:.3e}"
                    )
                if args.skip_nonfinite_batches:
                    continue
                raise RuntimeError(f"Non-finite params at step {step}.")

            last_good_state = snapshot_trainable_state()

            rec = {
                "step": step,
                "loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": grad_norm,
            }
            log_f.write(json.dumps(rec) + "\n")
            if step % max(1, args.log_every) == 0 or step == 1:
                print(
                    f"[train] step={step} "
                    f"loss={rec['loss']:.6f} grad_norm={grad_norm:.6f}"
                )

            if args.eval_every is not None and step % max(1, args.eval_every) == 0:
                op_eval = _operator_sanity_check(
                    model=student.eval(),
                    env=args.env,
                    device=device,
                )
                student.train()
                if not op_eval["finite_student_costs"]:
                    nonfinite_operator_eval_count += 1
                rec_eval = {
                    "step": step,
                    "operator_eval": op_eval,
                }
                log_f.write(json.dumps(rec_eval) + "\n")
                if op_eval["passed"] and float(loss.item()) < best_valid_loss:
                    best_valid_loss = float(loss.item())
                    best_valid_state = snapshot_trainable_state()
                    best_valid_operator = op_eval

    if best_valid_state is not None:
        restore_trainable_state(best_valid_state)

    final_operator = _operator_sanity_check(
        model=student.eval(),
        env=args.env,
        device=device,
    )
    run_success = bool(final_operator["passed"])
    if not run_success and args.restore_best_on_failure and best_valid_state is not None:
        restore_trainable_state(best_valid_state)
        final_operator = _operator_sanity_check(
            model=student.eval(),
            env=args.env,
            device=device,
        )
        run_success = bool(final_operator["passed"])

    student = student.to("cpu").eval()
    student.requires_grad_(False)
    distilled_path = out_dir / "distilled_model.pt"
    if run_success:
        torch.save(student, distilled_path)

    inherited = load_inherited_compression_report(args.student_path)
    baseline_metrics_path, baseline_metrics = _load_baseline_metrics(
        args.student_path
    )
    top5_block = (
        baseline_metrics.get("topk_overlap", {}).get("5", {})
        if baseline_metrics is not None
        else {}
    )
    method_status = "valid" if run_success else "invalid_operator_unstable"
    if initial_loss is None:
        method_status = "training_failed"

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "prediction_only_distillation",
        "env": args.env,
        "teacher_family": args.teacher_family,
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_init": str(args.student_path),
        "tag": args.tag,
        "dataset_source": args.dataset_source,
        "max_steps": int(max_steps),
        "batch_size": int(args.batch_size),
        "horizon": int(args.horizon),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "trainable_substring": args.trainable_substring,
        "num_trainable_params": int(num_trainable),
        "num_frozen_params": int(num_frozen),
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "loss_decreased": bool(
            initial_loss is not None
            and final_loss is not None
            and final_loss < initial_loss
        ),
        "first_step_loss_requires_grad": first_step_loss_requires_grad,
        "first_step_nonzero_grad_params": int(first_step_nonzero_grad_params),
        "first_step_total_grad_norm": float(first_step_total_grad_norm),
        "distilled_model_path": str(distilled_path) if run_success else None,
        "has_get_cost_after_distillation": bool(hasattr(student, "get_cost")),
        "run_success": run_success,
        "skip_nonfinite_batches": bool(args.skip_nonfinite_batches),
        "skipped_nonfinite_batches": int(skipped_nonfinite_batches),
        "nonfinite_operator_eval_count": int(nonfinite_operator_eval_count),
        "save_best_by": args.save_best_by,
        "restore_best_on_failure": bool(args.restore_best_on_failure),
        "best_valid_loss": (
            float(best_valid_loss) if best_valid_state is not None else None
        ),
        "best_valid_operator_eval": best_valid_operator,
        "final_operator_validation": final_operator,
        "prediction_target_sanity": {
            "teacher_tensor": "teacher.rollout(...)[\"predicted_emb\"]",
            "student_tensor": "student.rollout(...)[\"predicted_emb\"]",
            "shape": first_pred_shape,
            "same_latent_space_claim": (
                "Both tensors are produced by each model's rollout "
                "predictor pathway before cost criterion."
            ),
            "uncertain_prediction_target": False,
        },
        "method_status": method_status,
        "failure_interpretation": (
            "Prediction-only latent MSE recovery produced non-finite or "
            "collapsed planner costs; no deployable distilled model saved."
            if not run_success
            else ""
        ),
        "baseline_metrics_path": baseline_metrics_path,
        "baseline_finite_student_costs": (
            baseline_metrics.get("finite_student_costs")
            if baseline_metrics is not None
            else None
        ),
        "baseline_spearman_mean": (
            baseline_metrics.get("spearman_per_state", {}).get("mean")
            if baseline_metrics is not None
            else None
        ),
        "baseline_top5_overlap_mean": top5_block.get("mean"),
        "baseline_teacher_regret_mean": (
            baseline_metrics.get("teacher_regret", {}).get("mean")
            if baseline_metrics is not None
            else None
        ),
        "distilled_valid_for_comparison": bool(run_success),
        "inherited_compression": inherited,
    }
    save_json(out_dir / "distillation_report.json", report)

    if run_success:
        print("Prediction-only distillation complete")
    else:
        print("Prediction-only distillation failed validation")
    print(f"  tag:                        {args.tag}")
    print(f"  initial loss:               {initial_loss:.6f}")
    print(f"  final loss:                 {final_loss:.6f}")
    print(f"  loss decreased:             {report['loss_decreased']}")
    print(f"  first-step grad params >0:  {first_step_nonzero_grad_params}")
    print(f"  first-step grad norm:       {first_step_total_grad_norm:.6f}")
    print(f"  operator validation passed: {run_success}")
    print(f"  distilled model:            {run_success and distilled_path or 'not saved'}")

    if not run_success:
        raise RuntimeError(
            "Final operator validation failed. "
            "distilled_model.pt was not saved."
        )


if __name__ == "__main__":
    main()
