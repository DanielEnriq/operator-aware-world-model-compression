from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-elite", type=float, default=1.0)
    parser.add_argument("--lambda-pred", type=float, default=0.0)
    parser.add_argument("--tau-kl", type=float, default=1.0)
    parser.add_argument("--tau-elite", type=float, default=1.0)
    parser.add_argument("--elite-k", type=int, default=10)
    parser.add_argument("--elite-frac", type=float, default=None)
    parser.add_argument(
        "--elite-loss",
        default="balanced_bce",
        choices=["balanced_bce", "bce"],
    )
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument(
        "--save-best-by",
        default="hybrid_operator_score",
        choices=["hybrid_operator_score"],
    )
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
    student_probs, student_log_probs = _distribution_from_costs(
        student_costs,
        tau=tau,
        normalize_costs=normalize_costs,
        eps=eps,
    )
    kl_per_state = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    return kl_per_state.mean(), teacher_probs, student_probs


def _resolve_elite_k(
    num_candidates: int,
    elite_k: int,
    elite_frac: float | None,
) -> int:
    if elite_frac is not None:
        k = int(elite_frac * num_candidates)
    else:
        k = int(elite_k)
    k = max(1, min(k, num_candidates - 1))
    return k


def _elite_loss(
    teacher_costs: torch.Tensor,
    student_costs: torch.Tensor,
    *,
    elite_k: int,
    tau: float,
    normalize_costs: str,
    eps: float,
    loss_mode: str,
) -> torch.Tensor:
    c_s = _normalize_costs(student_costs, normalize_costs, eps)
    logits = -c_s / tau
    topk_idx = torch.topk(
        teacher_costs,
        k=elite_k,
        dim=-1,
        largest=False,
    ).indices
    labels = torch.zeros_like(logits)
    labels.scatter_(1, topk_idx, 1.0)

    if loss_mode == "bce":
        n = logits.shape[1]
        pos_weight_val = float((n - elite_k) / max(1, elite_k))
        pos_weight = torch.tensor(
            pos_weight_val,
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
        )

    pos_mask = labels > 0.5
    neg_mask = ~pos_mask
    pos_losses = F.binary_cross_entropy_with_logits(
        logits[pos_mask],
        labels[pos_mask],
        reduction="mean",
    )
    neg_losses = F.binary_cross_entropy_with_logits(
        logits[neg_mask],
        labels[neg_mask],
        reduction="mean",
    )
    return pos_losses + neg_losses


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


def _slice_info_dict(
    info: dict[str, torch.Tensor],
    idx: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in info.items()}


def _operator_metrics(
    teacher_costs: torch.Tensor,
    student_costs: torch.Tensor,
    teacher_best_first_action: torch.Tensor,
    candidate_actions: torch.Tensor,
) -> dict:
    finite = bool(torch.isfinite(student_costs).all().item())
    if not finite:
        return {
            "finite_student_costs": False,
            "student_cost_std": float("nan"),
            "spearman_mean": float("nan"),
            "top5_overlap_mean": float("nan"),
            "top10_overlap_mean": float("nan"),
            "teacher_regret_mean": float("nan"),
            "first_action_error_mean": float("nan"),
            "teacher_best_index_match_rate": 0.0,
            "student_best_constant": False,
        }

    t_sorted = torch.argsort(teacher_costs, dim=1)
    s_sorted = torch.argsort(student_costs, dim=1)
    overlaps5, overlaps10, spearman = [], [], []
    for i in range(teacher_costs.shape[0]):
        t5, s5 = set(t_sorted[i, :5].tolist()), set(s_sorted[i, :5].tolist())
        t10, s10 = set(t_sorted[i, :10].tolist()), set(s_sorted[i, :10].tolist())
        overlaps5.append(len(t5.intersection(s5)) / 5.0)
        overlaps10.append(len(t10.intersection(s10)) / 10.0)
        spearman.append(_spearman(teacher_costs[i], student_costs[i]))

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
    student_best_first_action = candidate_actions[
        torch.arange(candidate_actions.shape[0], device=candidate_actions.device),
        student_best,
        0,
        :,
    ]
    first_action_error = torch.linalg.norm(
        student_best_first_action - teacher_best_first_action,
        dim=1,
    )
    unique_student_best = int(torch.unique(student_best).numel())

    return {
        "finite_student_costs": True,
        "student_cost_std": float(student_costs.std().item()),
        "spearman_mean": float(
            torch.tensor(spearman, device=teacher_costs.device).mean().item()
        ),
        "top5_overlap_mean": float(
            torch.tensor(overlaps5, device=teacher_costs.device).mean().item()
        ),
        "top10_overlap_mean": float(
            torch.tensor(overlaps10, device=teacher_costs.device).mean().item()
        ),
        "teacher_regret_mean": float(regret.mean().item()),
        "first_action_error_mean": float(first_action_error.mean().item()),
        "teacher_best_index_match_rate": float(
            (teacher_best == student_best).float().mean().item()
        ),
        "student_best_constant": bool(
            unique_student_best == 1 and teacher_costs.shape[0] > 1
        ),
    }


def _entropy_mean(probs: torch.Tensor, eps: float) -> float:
    return float((-(probs * torch.log(probs + eps)).sum(dim=-1).mean()).item())


def _hybrid_validation_score(metrics: dict) -> float:
    return float(
        -metrics["teacher_regret_mean"]
        + 50.0 * metrics["top10_overlap_mean"]
        + 10.0 * metrics["spearman_mean"]
    )


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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    prediction_loss_requested = bool(args.lambda_pred > 0.0)
    prediction_loss_active = False

    out_dir = Path(args.output_root) / args.env / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    train_cache_path, eval_cache_path = _resolve_cache_paths(args)
    train_cache = torch.load(train_cache_path, map_location="cpu", weights_only=False)
    eval_cache = torch.load(eval_cache_path, map_location="cpu", weights_only=False)

    train_candidate_actions = train_cache["candidate_actions"].float()
    train_teacher_costs = train_cache["teacher_costs"].float()
    train_teacher_best_first_action = train_cache["teacher_best_first_action"].float()
    train_episodes_idx = list(train_cache["episodes_idx"])
    train_start_steps = list(train_cache["start_steps"])
    train_goal_offset_steps = int(train_cache["goal_offset_steps"])

    eval_candidate_actions = eval_cache["candidate_actions"].float()
    eval_teacher_costs = eval_cache["teacher_costs"].float()
    eval_teacher_best_first_action = eval_cache["teacher_best_first_action"].float()
    eval_episodes_idx = list(eval_cache["episodes_idx"])
    eval_start_steps = list(eval_cache["start_steps"])
    eval_goal_offset_steps = int(eval_cache["goal_offset_steps"])

    if args.limit_states is not None:
        s = int(args.limit_states)
        train_candidate_actions = train_candidate_actions[:s]
        train_teacher_costs = train_teacher_costs[:s]
        train_teacher_best_first_action = train_teacher_best_first_action[:s]
        train_episodes_idx = train_episodes_idx[:s]
        train_start_steps = train_start_steps[:s]
        eval_candidate_actions = eval_candidate_actions[:s]
        eval_teacher_costs = eval_teacher_costs[:s]
        eval_teacher_best_first_action = eval_teacher_best_first_action[:s]
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
    elite_k = _resolve_elite_k(
        train_n_candidates,
        args.elite_k,
        args.elite_frac,
    )

    student = load_model_from_path(args.student_path, device=device)
    student.eval()
    trainable_params = set_trainable_by_substring(student, args.trainable_substring)
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
    train_teacher_best_first_action = train_teacher_best_first_action.to(device)

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
    eval_candidate_actions_device = eval_candidate_actions.to(device)
    eval_teacher_costs = eval_teacher_costs.to(device)
    eval_teacher_best_first_action = eval_teacher_best_first_action.to(device)

    val_idx_np = np.arange(min(4, eval_n_states), dtype=np.int64)
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

    initial_total_loss = None
    final_total_loss = None
    initial_kl_loss = None
    final_kl_loss = None
    initial_elite_loss = None
    final_elite_loss = None
    initial_pred_loss = None
    final_pred_loss = None

    best_state = None
    best_step = None
    best_validation_score = -float("inf")
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
            costs_require_grad = bool(student_costs.requires_grad)

            kl_loss, teacher_probs, student_probs = _kl_from_costs(
                t_cost_batch,
                student_costs,
                tau=float(args.tau_kl),
                normalize_costs=args.normalize_costs,
                eps=float(args.eps),
            )
            elite_loss = _elite_loss(
                t_cost_batch,
                student_costs,
                elite_k=elite_k,
                tau=float(args.tau_elite),
                normalize_costs=args.normalize_costs,
                eps=float(args.eps),
                loss_mode=args.elite_loss,
            )
            pred_loss = torch.tensor(0.0, device=device)
            pred_loss_used = False
            # Optional and intentionally disabled by default.
            if prediction_loss_requested:
                pred_loss = torch.tensor(float("nan"), device=device)
                if torch.isfinite(pred_loss):
                    pred_loss_used = True
                    prediction_loss_active = True

            total_loss = (
                float(args.lambda_kl) * kl_loss
                + float(args.lambda_elite) * elite_loss
            )
            if prediction_loss_active and pred_loss_used:
                total_loss = total_loss + float(args.lambda_pred) * pred_loss

            kl_val = float(kl_loss.item()) if torch.isfinite(kl_loss) else float("nan")
            elite_val = (
                float(elite_loss.item()) if torch.isfinite(elite_loss) else float("nan")
            )
            total_val = (
                float(total_loss.item())
                if torch.isfinite(total_loss)
                else float("nan")
            )
            pred_val = (
                float(pred_loss.item())
                if pred_loss_used and torch.isfinite(pred_loss)
                else None
            )

            if initial_total_loss is None and np.isfinite(total_val):
                initial_total_loss = total_val
            if np.isfinite(total_val):
                final_total_loss = total_val
            if initial_kl_loss is None and np.isfinite(kl_val):
                initial_kl_loss = kl_val
            if np.isfinite(kl_val):
                final_kl_loss = kl_val
            if initial_elite_loss is None and np.isfinite(elite_val):
                initial_elite_loss = elite_val
            if np.isfinite(elite_val):
                final_elite_loss = elite_val
            if pred_val is not None and initial_pred_loss is None:
                initial_pred_loss = pred_val
            if pred_val is not None:
                final_pred_loss = pred_val

            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(total_loss):
                total_loss.backward()

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
                first_step_loss_requires_grad = bool(total_loss.requires_grad)
                first_step_nonzero_grad_params = int(nonzero)
                first_step_total_grad_norm = grad_norm

            if torch.isfinite(total_loss):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()

            step_time = time.time() - step_start
            rec = {
                "step": step,
                "total_loss": total_val,
                "kl_loss": kl_val,
                "elite_loss": elite_val,
                "pred_loss": pred_val,
                "grad_norm": grad_norm,
                "student_cost_std": float(student_costs.detach().std().item()),
                "teacher_entropy": _entropy_mean(teacher_probs, float(args.eps)),
                "student_entropy": _entropy_mean(student_probs, float(args.eps)),
                "step_time_sec": float(step_time),
                "candidate_actions_shape": list(cand_batch.shape),
                "student_costs_shape": list(student_costs.shape),
                "student_costs_requires_grad": costs_require_grad,
                "total_loss_requires_grad": bool(total_loss.requires_grad),
            }
            log_f.write(json.dumps(rec) + "\n")

            if step % max(1, args.log_every) == 0 or step == 1:
                print(
                    f"[train] step={step} total={total_val:.6f} "
                    f"kl={kl_val:.6f} elite={elite_val:.6f} "
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
                    teacher_best_val = eval_teacher_best_first_action[val_idx]
                    exp_val = expand_info_for_candidates(
                        info_val,
                        num_candidates=eval_n_candidates,
                    )
                    s_cost_val = compute_model_costs(student, exp_val, cand_val)

                    val_kl_t, _, _ = _kl_from_costs(
                        t_cost_val,
                        s_cost_val,
                        tau=float(args.tau_kl),
                        normalize_costs=args.normalize_costs,
                        eps=float(args.eps),
                    )
                    val_elite_t = _elite_loss(
                        t_cost_val,
                        s_cost_val,
                        elite_k=elite_k,
                        tau=float(args.tau_elite),
                        normalize_costs=args.normalize_costs,
                        eps=float(args.eps),
                        loss_mode=args.elite_loss,
                    )
                    val_kl = (
                        float(val_kl_t.item())
                        if torch.isfinite(val_kl_t)
                        else float("nan")
                    )
                    val_elite = (
                        float(val_elite_t.item())
                        if torch.isfinite(val_elite_t)
                        else float("nan")
                    )
                    cand_val_raw = eval_candidate_actions_device[val_idx]
                    val_metrics = _operator_metrics(
                        t_cost_val,
                        s_cost_val,
                        teacher_best_val,
                        cand_val_raw,
                    )
                    val_ok = bool(
                        np.isfinite(val_kl)
                        and np.isfinite(val_elite)
                        and val_metrics["finite_student_costs"]
                        and val_metrics["student_cost_std"] > 1e-6
                        and not val_metrics["student_best_constant"]
                    )
                    score = (
                        _hybrid_validation_score(val_metrics)
                        if val_ok
                        else -float("inf")
                    )

                log_f.write(
                    json.dumps(
                        {
                            "step": step,
                            "validation_kl": val_kl,
                            "validation_elite_loss": val_elite,
                            "validation_metrics": val_metrics,
                            "validation_score": score,
                            "validation_passed": val_ok,
                        }
                    )
                    + "\n"
                )

                if val_ok and score > best_validation_score:
                    best_validation_score = score
                    best_step = int(step)
                    best_state = snapshot()

    if best_state is not None:
        restore(best_state)

    with torch.no_grad():
        val_idx = torch.as_tensor(val_idx_np, dtype=torch.long, device=device)
        info_val = _slice_info_dict(eval_info_all, val_idx)
        cand_val = eval_cand_eval_all[val_idx]
        t_cost_val = eval_teacher_costs[val_idx]
        teacher_best_val = eval_teacher_best_first_action[val_idx]
        exp_val = expand_info_for_candidates(
            info_val,
            num_candidates=eval_n_candidates,
        )
        s_cost_val = compute_model_costs(student, exp_val, cand_val)
        final_kl_t, _, _ = _kl_from_costs(
            t_cost_val,
            s_cost_val,
            tau=float(args.tau_kl),
            normalize_costs=args.normalize_costs,
            eps=float(args.eps),
        )
        final_elite_t = _elite_loss(
            t_cost_val,
            s_cost_val,
            elite_k=elite_k,
            tau=float(args.tau_elite),
            normalize_costs=args.normalize_costs,
            eps=float(args.eps),
            loss_mode=args.elite_loss,
        )
        final_kl = (
            float(final_kl_t.item()) if torch.isfinite(final_kl_t) else float("nan")
        )
        final_elite = (
            float(final_elite_t.item())
            if torch.isfinite(final_elite_t)
            else float("nan")
        )
        cand_val_raw = eval_candidate_actions_device[val_idx]
        final_operator_validation = _operator_metrics(
            t_cost_val,
            s_cost_val,
            teacher_best_val,
            cand_val_raw,
        )
        run_success = bool(
            np.isfinite(final_kl)
            and np.isfinite(final_elite)
            and final_operator_validation["finite_student_costs"]
            and final_operator_validation["student_cost_std"] > 1e-6
            and not final_operator_validation["student_best_constant"]
        )

    method_status = "valid" if run_success else "invalid_operator_unstable"
    if initial_total_loss is None:
        method_status = "training_failed"

    student_cpu = student.to("cpu").eval()
    student_cpu.requires_grad_(False)
    distilled_path = out_dir / "distilled_model.pt"
    if run_success:
        torch.save(student_cpu, distilled_path)

    inherited = load_inherited_compression_report(args.student_path)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "operator_hybrid_distillation",
        "method_status": method_status,
        "env": args.env,
        "teacher_cache": str(train_cache_path),
        "train_cache": str(train_cache_path),
        "eval_cache": str(eval_cache_path),
        "heldout_operator_validation": bool(train_cache_path != eval_cache_path),
        "student_init": str(args.student_path),
        "tag": args.tag,
        "lambda_kl": float(args.lambda_kl),
        "lambda_elite": float(args.lambda_elite),
        "lambda_pred": float(args.lambda_pred),
        "tau_kl": float(args.tau_kl),
        "tau_elite": float(args.tau_elite),
        "elite_k": int(elite_k),
        "elite_frac": args.elite_frac,
        "elite_loss": args.elite_loss,
        "normalize_costs": args.normalize_costs,
        "trainable_substring": args.trainable_substring,
        "max_steps": int(args.max_steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "num_trainable_params": int(num_trainable),
        "num_frozen_params": int(num_frozen),
        "first_step_loss_requires_grad": first_step_loss_requires_grad,
        "first_step_nonzero_grad_params": int(first_step_nonzero_grad_params),
        "first_step_total_grad_norm": float(first_step_total_grad_norm),
        "initial_total_loss": initial_total_loss,
        "final_total_loss": final_total_loss,
        "initial_kl_loss": initial_kl_loss,
        "final_kl_loss": final_kl_loss,
        "initial_elite_loss": initial_elite_loss,
        "final_elite_loss": final_elite_loss,
        "prediction_loss_active": prediction_loss_active,
        "initial_prediction_loss": initial_pred_loss,
        "final_prediction_loss": final_pred_loss,
        "best_step": best_step,
        "best_validation_score": (
            float(best_validation_score)
            if np.isfinite(best_validation_score)
            else None
        ),
        "final_operator_validation": final_operator_validation,
        "run_success": run_success,
        "distilled_model_path": str(distilled_path) if run_success else None,
        "inherited_compression": inherited,
    }
    save_json(out_dir / "distillation_report.json", report)

    print("Operator hybrid distillation finished")
    print(f"  tag:                   {args.tag}")
    print(f"  run_success:           {run_success}")
    print(f"  first-step grad ok:    {first_step_nonzero_grad_params > 0}")
    print(f"  first-step requiresgrad:{first_step_loss_requires_grad}")
    print(f"  initial/final kl:      {initial_kl_loss} -> {final_kl_loss}")
    print(
        "  initial/final elite:   "
        f"{initial_elite_loss} -> {final_elite_loss}"
    )
    print(f"  best validation score: {report['best_validation_score']}")
    print(
        "  distilled model:       "
        f"{run_success and distilled_path or 'not saved'}"
    )
    print("  step behavior: cache-only cost forward, no CEM/world.")


if __name__ == "__main__":
    main()
