from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

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
    load_inherited_compression_report,
    predictor_param_partition,
    set_trainable_by_substring,
)
from oawc.compression.reports import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument(
        "--teacher-cache",
        default=None,
        help="Backward-compatible alias for --train-cache.",
    )
    parser.add_argument("--train-cache", default=None)
    parser.add_argument("--eval-cache", default=None)
    parser.add_argument("--student-path", required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--normalize-costs",
        default="zscore",
        choices=["none", "zscore"],
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--trainable-substring", default="predictor")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-states", type=int, default=64)
    parser.add_argument("--val-candidates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument(
        "--save-best-by",
        default="val_loss",
        choices=["val_spearman", "val_top5", "val_regret", "val_loss"],
    )
    parser.add_argument(
        "--early-stop-metric",
        default=None,
        choices=["spearman", "top5", "regret", "loss"],
    )
    parser.add_argument("--save-best-by-val", action="store_true")
    parser.add_argument("--restore-best-on-failure", action="store_true")
    parser.add_argument(
        "--no-restore-best-on-failure",
        dest="restore_best_on_failure",
        action="store_false",
    )
    parser.set_defaults(restore_best_on_failure=True)
    parser.add_argument("--limit-states", type=int, default=None)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-root", default="outputs/compression")
    return parser.parse_args()


def _normalize_costs(
    c: torch.Tensor,
    mode: str,
    eps: float,
) -> torch.Tensor:
    if mode == "none":
        return c
    mu = c.mean(dim=-1, keepdim=True)
    std = c.std(dim=-1, keepdim=True)
    std = torch.where(std < eps, torch.full_like(std, eps), std)
    return (c - mu) / std


def _distribution_from_costs(
    costs: torch.Tensor,
    *,
    tau: float,
    normalize_costs: str,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = _normalize_costs(costs, normalize_costs, eps)
    log_probs = torch.log_softmax(-c / tau, dim=-1)
    probs = log_probs.exp()
    return probs, log_probs


def _kl_from_costs(
    teacher_costs: torch.Tensor,
    student_costs: torch.Tensor,
    *,
    tau: float,
    normalize_costs: str,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_probs, teacher_log_probs = _distribution_from_costs(
        teacher_costs,
        tau=tau,
        normalize_costs=normalize_costs,
        eps=eps,
    )
    _, student_log_probs = _distribution_from_costs(
        student_costs,
        tau=tau,
        normalize_costs=normalize_costs,
        eps=eps,
    )
    kl_per_state = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    return kl_per_state.mean(), teacher_probs, student_log_probs.exp()


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    idx_x = torch.argsort(x)
    idx_y = torch.argsort(y)
    rx = torch.empty_like(idx_x, dtype=torch.float32)
    ry = torch.empty_like(idx_y, dtype=torch.float32)
    rx[idx_x] = torch.arange(
        x.numel(),
        dtype=torch.float32,
        device=x.device,
    )
    ry[idx_y] = torch.arange(
        y.numel(),
        dtype=torch.float32,
        device=y.device,
    )
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx.pow(2).sum()) * (ry.pow(2).sum()))
    if float(denom.item()) == 0.0:
        return 0.0
    return float((rx * ry).sum().item() / denom.item())


def _operator_metrics(
    teacher_costs: torch.Tensor,
    student_costs: torch.Tensor,
) -> dict:
    finite = bool(torch.isfinite(student_costs).all().item())
    if not finite:
        return {
            "finite_student_costs": False,
            "student_cost_std": float("nan"),
            "spearman_mean": float("nan"),
            "top5_overlap_mean": float("nan"),
            "teacher_regret_mean": float("nan"),
            "teacher_best_index_match_rate": 0.0,
            "student_best_constant": False,
        }

    student_std = float(student_costs.std().item())
    t_sorted = torch.argsort(teacher_costs, dim=1)
    s_sorted = torch.argsort(student_costs, dim=1)
    top5_overlap = []
    spearman = []
    for i in range(teacher_costs.shape[0]):
        t5 = set(t_sorted[i, :5].tolist())
        s5 = set(s_sorted[i, :5].tolist())
        top5_overlap.append(len(t5.intersection(s5)) / 5.0)
        spearman.append(_spearman(teacher_costs[i], student_costs[i]))
    metrics_device = teacher_costs.device
    top5_overlap_mean = float(
        torch.tensor(top5_overlap, device=metrics_device).mean().item()
    )
    spearman_mean = float(
        torch.tensor(spearman, device=metrics_device).mean().item()
    )

    teacher_best = t_sorted[:, 0]
    student_best = s_sorted[:, 0]
    teacher_best_cost = teacher_costs[
        torch.arange(teacher_costs.shape[0], device=teacher_costs.device),
        teacher_best,
    ]
    student_pick_teacher_cost = teacher_costs[
        torch.arange(teacher_costs.shape[0], device=teacher_costs.device),
        student_best,
    ]
    regret = student_pick_teacher_cost - teacher_best_cost
    match_rate = float((teacher_best == student_best).float().mean().item())
    unique_student_best = int(torch.unique(student_best).numel())

    return {
        "finite_student_costs": True,
        "student_cost_std": student_std,
        "spearman_mean": spearman_mean,
        "top5_overlap_mean": top5_overlap_mean,
        "teacher_regret_mean": float(regret.mean().item()),
        "teacher_best_index_match_rate": match_rate,
        "student_best_constant": bool(
            unique_student_best == 1 and teacher_costs.shape[0] > 1
        ),
    }


def _slice_info_dict(
    info: dict[str, torch.Tensor],
    idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in info.items():
        out[k] = v[idx]
    return out


def _resolve_cache_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    train_cache_arg = args.train_cache or args.teacher_cache
    if train_cache_arg is None:
        raise ValueError("Provide --train-cache (or legacy --teacher-cache).")
    train_cache = Path(train_cache_arg)
    eval_cache = Path(args.eval_cache) if args.eval_cache else train_cache
    if args.eval_cache is None:
        print(
            "[warn] --eval-cache not provided; using train cache for "
            "validation. For held-out protocol, pass --eval-cache."
        )
    return train_cache, eval_cache


def _is_better_checkpoint(
    *,
    save_best_by: str,
    candidate_metrics: dict,
    candidate_loss: float,
    best_metrics: dict | None,
    best_loss: float | None,
) -> bool:
    if best_metrics is None or best_loss is None:
        return True
    if save_best_by == "val_spearman":
        return float(candidate_metrics["spearman_mean"]) > float(
            best_metrics["spearman_mean"]
        )
    if save_best_by == "val_top5":
        return float(candidate_metrics["top5_overlap_mean"]) > float(
            best_metrics["top5_overlap_mean"]
        )
    if save_best_by == "val_regret":
        return float(candidate_metrics["teacher_regret_mean"]) < float(
            best_metrics["teacher_regret_mean"]
        )
    return float(candidate_loss) < float(best_loss)


def main() -> None:
    args = parse_args()
    if args.early_stop_metric is not None:
        args.save_best_by = {
            "spearman": "val_spearman",
            "top5": "val_top5",
            "regret": "val_regret",
            "loss": "val_loss",
        }[str(args.early_stop_metric)]
    if args.save_best_by_val and args.early_stop_metric is None:
        args.save_best_by = "val_top5"
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    run_start = time.time()

    out_dir = Path(args.output_root) / args.env / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    train_cache_path, eval_cache_path = _resolve_cache_paths(args)
    train_cache = torch.load(train_cache_path, map_location="cpu", weights_only=False)
    eval_cache = torch.load(eval_cache_path, map_location="cpu", weights_only=False)

    train_candidate_actions = train_cache["candidate_actions"].float()
    train_teacher_costs = train_cache["teacher_costs"].float()
    train_episodes_idx = list(train_cache["episodes_idx"])
    train_start_steps = list(train_cache["start_steps"])
    train_goal_offset_steps = int(train_cache["goal_offset_steps"])

    eval_candidate_actions = eval_cache["candidate_actions"].float()
    eval_teacher_costs = eval_cache["teacher_costs"].float()
    eval_episodes_idx = list(eval_cache["episodes_idx"])
    eval_start_steps = list(eval_cache["start_steps"])
    eval_goal_offset_steps = int(eval_cache["goal_offset_steps"])

    if args.limit_states is not None:
        s = int(args.limit_states)
        train_candidate_actions = train_candidate_actions[:s]
        train_teacher_costs = train_teacher_costs[:s]
        train_episodes_idx = train_episodes_idx[:s]
        train_start_steps = train_start_steps[:s]
        eval_candidate_actions = eval_candidate_actions[:s]
        eval_teacher_costs = eval_teacher_costs[:s]
        eval_episodes_idx = eval_episodes_idx[:s]
        eval_start_steps = eval_start_steps[:s]
    if args.limit_candidates is not None:
        c = int(args.limit_candidates)
        train_candidate_actions = train_candidate_actions[:, :c]
        train_teacher_costs = train_teacher_costs[:, :c]
        eval_candidate_actions = eval_candidate_actions[:, :c]
        eval_teacher_costs = eval_teacher_costs[:, :c]

    train_n_states = int(train_candidate_actions.shape[0])
    train_n_candidates = int(train_candidate_actions.shape[1])
    eval_n_states = int(eval_candidate_actions.shape[0])
    eval_n_candidates = int(eval_candidate_actions.shape[1])

    student = load_model_from_path(args.student_path, device=device)
    # Keep eval mode to avoid BatchNorm single-sample instability while still
    # allowing gradients through trainable predictor parameters.
    student.eval()
    trainable_params = set_trainable_by_substring(
        student,
        args.trainable_substring,
    )
    if not trainable_params:
        raise ValueError(
            f"No trainable params matched substring={args.trainable_substring!r}"
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

    train_info_all = build_info_dict_from_cache(
        env_name=args.env,
        episodes_idx=train_episodes_idx,
        start_steps=train_start_steps,
        goal_offset_steps=train_goal_offset_steps,
        device=device,
    )
    train_info_all = maybe_align_action_width(train_info_all, student)
    train_cand_eval_all = adapt_candidates_for_model(
        train_candidate_actions.to(device),
        student,
    )
    train_teacher_costs = train_teacher_costs.to(device)

    eval_info_all = build_info_dict_from_cache(
        env_name=args.env,
        episodes_idx=eval_episodes_idx,
        start_steps=eval_start_steps,
        goal_offset_steps=eval_goal_offset_steps,
        device=device,
    )
    eval_info_all = maybe_align_action_width(eval_info_all, student)
    eval_cand_eval_all = adapt_candidates_for_model(
        eval_candidate_actions.to(device),
        student,
    )
    eval_teacher_costs = eval_teacher_costs.to(device)

    val_count = max(1, min(int(args.val_states), eval_n_states))
    val_idx_np = np.arange(val_count, dtype=np.int64)
    if len(val_idx_np) == 0:
        raise ValueError("No states available in eval cache.")
    train_idx_np = np.arange(train_n_states, dtype=np.int64)

    def snapshot() -> dict[str, torch.Tensor]:
        return {
            name: p.detach().cpu().clone()
            for name, p in student.named_parameters()
            if p.requires_grad
        }

    def restore(state: dict[str, torch.Tensor]) -> None:
        named = dict(student.named_parameters())
        for name, t in state.items():
            named[name].data.copy_(t.to(named[name].device))

    first_step_loss_requires_grad = False
    first_step_nonzero_grad_params = 0
    first_step_total_grad_norm = 0.0
    initial_train_kl = None
    final_train_kl = None
    teacher_entropy_mean = None
    final_student_entropy_mean = None

    best_state = None
    best_step = None
    best_validation_kl = float("inf")
    best_validation_metrics = None
    run_success = False
    final_operator_validation = None

    with log_path.open("w", encoding="utf-8") as log_f:
        for step in range(1, int(args.max_steps) + 1):
            batch_idx_np = rng.choice(
                train_idx_np,
                size=min(args.batch_size, len(train_idx_np)),
                replace=True,
            )
            batch_idx = torch.as_tensor(
                batch_idx_np,
                dtype=torch.long,
                device=device,
            )
            info_batch = _slice_info_dict(train_info_all, batch_idx)
            cand_batch = train_cand_eval_all[batch_idx]
            t_cost_batch = train_teacher_costs[batch_idx]
            expanded = expand_info_for_candidates(
                info_batch,
                num_candidates=train_n_candidates,
            )

            step_start = time.time()
            student_costs = compute_model_costs(student, expanded, cand_batch)
            requires_grad = bool(student_costs.requires_grad)
            kl, teacher_probs, student_probs = _kl_from_costs(
                t_cost_batch,
                student_costs,
                tau=float(args.tau),
                normalize_costs=args.normalize_costs,
                eps=float(args.eps),
            )
            train_kl = float(kl.item()) if torch.isfinite(kl) else float("nan")
            teacher_entropy = float(
                (
                    -(teacher_probs * torch.log(teacher_probs + args.eps))
                    .sum(dim=-1)
                    .mean()
                ).item()
            )
            student_entropy = float(
                (
                    -(student_probs * torch.log(student_probs + args.eps))
                    .sum(dim=-1)
                    .mean()
                ).item()
            )
            student_std = float(student_costs.detach().std().item())

            if initial_train_kl is None and np.isfinite(train_kl):
                initial_train_kl = train_kl
            if np.isfinite(train_kl):
                final_train_kl = train_kl
            teacher_entropy_mean = teacher_entropy
            final_student_entropy_mean = student_entropy

            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(kl):
                kl.backward()
            nonzero = 0
            grad_sq = 0.0
            for p in trainable_params:
                if p.grad is None:
                    continue
                gn = float(p.grad.norm().item())
                if gn > 0:
                    nonzero += 1
                grad_sq += gn * gn
            grad_norm = float(grad_sq ** 0.5)
            if step == 1:
                first_step_loss_requires_grad = bool(kl.requires_grad)
                first_step_nonzero_grad_params = int(nonzero)
                first_step_total_grad_norm = grad_norm

            if torch.isfinite(kl):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()

            step_time = time.time() - step_start
            rec = {
                "step": step,
                "train_kl": train_kl,
                "grad_norm": grad_norm,
                "teacher_entropy": teacher_entropy,
                "student_entropy": student_entropy,
                "student_cost_std": student_std,
                "step_time_sec": float(step_time),
                "candidate_actions_shape": list(cand_batch.shape),
                "student_costs_shape": list(student_costs.shape),
                "student_costs_requires_grad": requires_grad,
            }
            log_f.write(json.dumps(rec) + "\n")
            if step % max(1, args.log_every) == 0 or step == 1:
                print(
                    f"[train] step={step} kl={train_kl:.6f} "
                    f"grad={grad_norm:.6f} t={step_time:.3f}s"
                )

            if step % max(1, args.eval_every) == 0:
                with torch.no_grad():
                    val_idx = torch.as_tensor(
                        val_idx_np,
                        dtype=torch.long,
                        device=device,
                    )
                    info_val = _slice_info_dict(eval_info_all, val_idx)
                    cand_val = eval_cand_eval_all[val_idx]
                    t_cost_val = eval_teacher_costs[val_idx]
                    if args.val_candidates is not None:
                        vc = int(min(args.val_candidates, eval_n_candidates))
                        cand_val = cand_val[:, :vc]
                        t_cost_val = t_cost_val[:, :vc]
                    exp_val = expand_info_for_candidates(
                        info_val,
                        num_candidates=int(cand_val.shape[1]),
                    )
                    s_cost_val = compute_model_costs(
                        student,
                        exp_val,
                        cand_val,
                    )
                    val_kl_t, _, _ = _kl_from_costs(
                        t_cost_val,
                        s_cost_val,
                        tau=float(args.tau),
                        normalize_costs=args.normalize_costs,
                        eps=float(args.eps),
                    )
                    val_kl = (
                        float(val_kl_t.item())
                        if torch.isfinite(val_kl_t)
                        else float("nan")
                    )
                    val_metrics = _operator_metrics(t_cost_val, s_cost_val)
                    val_ok = bool(
                        np.isfinite(val_kl)
                        and val_metrics["finite_student_costs"]
                        and val_metrics["student_cost_std"] > 1e-6
                        and not val_metrics["student_best_constant"]
                    )
                rec_eval = {
                    "step": step,
                    "validation_kl": val_kl,
                    "validation_metrics": val_metrics,
                    "validation_passed": val_ok,
                }
                log_f.write(json.dumps(rec_eval) + "\n")
                if val_ok and _is_better_checkpoint(
                    save_best_by=args.save_best_by,
                    candidate_metrics=val_metrics,
                    candidate_loss=val_kl,
                    best_metrics=best_validation_metrics,
                    best_loss=best_validation_kl,
                ):
                    best_validation_kl = val_kl
                    best_validation_metrics = val_metrics
                    best_step = int(step)
                    best_state = snapshot()

    if best_state is not None:
        restore(best_state)

    with torch.no_grad():
        val_idx = torch.as_tensor(val_idx_np, dtype=torch.long, device=device)
        info_val = _slice_info_dict(eval_info_all, val_idx)
        cand_val = eval_cand_eval_all[val_idx]
        t_cost_val = eval_teacher_costs[val_idx]
        if args.val_candidates is not None:
            vc = int(min(args.val_candidates, eval_n_candidates))
            cand_val = cand_val[:, :vc]
            t_cost_val = t_cost_val[:, :vc]
        exp_val = expand_info_for_candidates(
            info_val,
            num_candidates=int(cand_val.shape[1]),
        )
        s_cost_val = compute_model_costs(student, exp_val, cand_val)
        val_kl_t, _, _ = _kl_from_costs(
            t_cost_val,
            s_cost_val,
            tau=float(args.tau),
            normalize_costs=args.normalize_costs,
            eps=float(args.eps),
        )
        val_kl = (
            float(val_kl_t.item())
            if torch.isfinite(val_kl_t)
            else float("nan")
        )
        final_operator_validation = _operator_metrics(t_cost_val, s_cost_val)
        run_success = bool(
            np.isfinite(val_kl)
            and final_operator_validation["finite_student_costs"]
            and final_operator_validation["student_cost_std"] > 1e-6
            and not final_operator_validation["student_best_constant"]
        )

    method_status = "valid" if run_success else "invalid_operator_unstable"
    if initial_train_kl is None:
        method_status = "training_failed"

    student_cpu = student.to("cpu").eval()
    student_cpu.requires_grad_(False)
    distilled_path = out_dir / "distilled_model.pt"
    if run_success:
        torch.save(student_cpu, distilled_path)

    inherited = load_inherited_compression_report(args.student_path)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "operator_cost_kl_distillation",
        "method_status": method_status,
        "env": args.env,
        "teacher_cache": str(train_cache_path),
        "train_cache": str(train_cache_path),
        "eval_cache": str(eval_cache_path),
        "heldout_operator_validation": bool(train_cache_path != eval_cache_path),
        "student_init": str(args.student_path),
        "tag": args.tag,
        "tau": float(args.tau),
        "normalize_costs": args.normalize_costs,
        "trainable_substring": args.trainable_substring,
        "max_steps": int(args.max_steps),
        "distill_steps": int(args.max_steps),
        "batch_size": int(args.batch_size),
        "train_states": int(train_n_states),
        "train_candidates": int(train_n_candidates),
        "val_states": int(val_count),
        "val_candidates": (
            int(args.val_candidates) if args.val_candidates is not None else None
        ),
        "save_best_by": args.save_best_by,
        "lr": float(args.lr),
        "optimizer": "AdamW",
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "num_trainable_params": int(num_trainable),
        "num_frozen_params": int(num_frozen),
        "first_step_loss_requires_grad": first_step_loss_requires_grad,
        "first_step_nonzero_grad_params": int(first_step_nonzero_grad_params),
        "first_step_total_grad_norm": float(first_step_total_grad_norm),
        "initial_train_kl": initial_train_kl,
        "final_train_kl": final_train_kl,
        "loss_decreased": bool(
            initial_train_kl is not None
            and final_train_kl is not None
            and final_train_kl < initial_train_kl
        ),
        "best_step": best_step,
        "best_validation_kl": (
            float(best_validation_kl)
            if np.isfinite(best_validation_kl)
            else None
        ),
        "final_operator_validation": final_operator_validation,
        "teacher_distribution_entropy_mean": teacher_entropy_mean,
        "final_student_distribution_entropy_mean": final_student_entropy_mean,
        "run_success": run_success,
        "wall_time_sec": float(time.time() - run_start),
        "approx_candidate_sequences_consumed": int(
            int(args.max_steps) * int(args.batch_size) * int(train_n_candidates)
        ),
        "teacher_labels_used": int(
            int(args.max_steps) * int(args.batch_size) * int(train_n_candidates)
        ),
        "validation_cache_used": str(eval_cache_path),
        "distilled_model_path": str(distilled_path) if run_success else None,
        "inherited_compression": inherited,
    }
    save_json(out_dir / "distillation_report.json", report)

    print("Operator cost-KL distillation finished")
    print(f"  tag:                   {args.tag}")
    print(f"  run_success:           {run_success}")
    print(f"  first-step grad ok:    {first_step_nonzero_grad_params > 0}")
    print(f"  first-step requiresgrad:{first_step_loss_requires_grad}")
    print(f"  initial/final kl:      {initial_train_kl} -> {final_train_kl}")
    print(f"  best val kl:           {report['best_validation_kl']}")
    print(
        "  distilled model:       "
        f"{run_success and distilled_path or 'not saved'}"
    )
    print("  step behavior: cache-only cost forward, no CEM/world.")


if __name__ == "__main__":
    main()
