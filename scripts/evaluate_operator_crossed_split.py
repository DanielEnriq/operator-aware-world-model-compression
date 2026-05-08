from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from oawc.compression.reports import save_json
from oawc.envs import ENV_SPECS
from oawc.models import load_cost_model


DEFAULT_MODEL_TAGS = [
    "lewm_tworoom_svd_r050",
    "lewm_tworoom_aa_svd_r050",
    "lewm_tworoom_svd_r050_cost_kl_split",
    "lewm_tworoom_svd_r050_elite_k10_split",
    "lewm_tworoom_svd_r050_hybrid_split",
    "lewm_tworoom_aa_svd_r025",
    "lewm_tworoom_aa_svd_r025_hybrid_split",
    "lewm_tworoom_svd_r025",
    "lewm_tworoom_svd_r025_hybrid_split",
]


@dataclass
class SplitCache:
    name: str
    path: Path
    payload: dict[str, Any]


@dataclass
class EvalCombo:
    state_split: str
    candidate_split: str
    candidate_mode: str
    info_cpu: dict[str, torch.Tensor]
    candidate_actions_cpu: torch.Tensor
    teacher_costs_cpu: torch.Tensor
    teacher_best_index: torch.Tensor
    teacher_best_first_action: torch.Tensor


def _stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1)
    return {
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "max": float(x.max().item()),
        "std": float(x.std().item()),
    }


def _sample_candidates(
    *,
    num_states: int,
    num_candidates: int,
    horizon: int,
    action_dim: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    actions = (
        torch.rand(
            (num_states, num_candidates, horizon, action_dim),
            generator=gen,
            dtype=torch.float32,
        )
        * 2.0
        - 1.0
    )
    return actions


def _slice_info_dict(
    info_dict: dict[str, torch.Tensor],
    state_slice: slice,
    device: str,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in info_dict.items():
        out[key] = value[state_slice].to(device)
    return out


def _compute_costs_chunked(
    *,
    model: torch.nn.Module,
    info_cpu: dict[str, torch.Tensor],
    candidate_actions_cpu: torch.Tensor,
    device: str,
    batch_states: int,
    batch_candidates: int,
    empty_cache_between_batches: bool,
) -> torch.Tensor:
    num_states = int(candidate_actions_cpu.shape[0])
    num_candidates = int(candidate_actions_cpu.shape[1])
    costs_cpu = torch.empty((num_states, num_candidates), dtype=torch.float32)
    for s0 in range(0, num_states, batch_states):
        s1 = min(num_states, s0 + batch_states)
        state_slice = slice(s0, s1)
        info_chunk = _slice_info_dict(info_cpu, state_slice, device)
        for c0 in range(0, num_candidates, batch_candidates):
            c1 = min(num_candidates, c0 + batch_candidates)
            cand_slice = slice(c0, c1)
            cand_chunk = candidate_actions_cpu[
                state_slice,
                cand_slice,
            ].to(device)
            cand_eval = adapt_candidates_for_model(cand_chunk, model)
            expanded = expand_info_for_candidates(
                info_chunk,
                num_candidates=int(c1 - c0),
            )
            with torch.no_grad():
                costs_chunk = compute_model_costs(model, expanded, cand_eval)
            costs_cpu[state_slice, cand_slice] = costs_chunk.detach().to("cpu")
            del cand_chunk
            del cand_eval
            del expanded
            del costs_chunk
            if (
                empty_cache_between_batches
                and device == "cuda"
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
        del info_chunk
    return costs_cpu


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    idx_x = torch.argsort(x)
    idx_y = torch.argsort(y)
    rx = torch.empty_like(idx_x, dtype=torch.float32)
    ry = torch.empty_like(idx_y, dtype=torch.float32)
    rx[idx_x] = torch.arange(x.numel(), dtype=torch.float32, device=x.device)
    ry[idx_y] = torch.arange(y.numel(), dtype=torch.float32, device=y.device)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.sqrt((rx.pow(2).sum()) * (ry.pow(2).sum()))
    if float(denom.item()) == 0.0:
        return 0.0
    return float((rx * ry).sum().item() / denom.item())


def _topk_overlap_mean(
    teacher_sorted: torch.Tensor,
    student_sorted: torch.Tensor,
    k: int,
) -> float:
    overlaps = []
    for i in range(teacher_sorted.shape[0]):
        t_set = set(teacher_sorted[i, :k].tolist())
        s_set = set(student_sorted[i, :k].tolist())
        overlaps.append(len(t_set.intersection(s_set)) / float(k))
    return float(
        torch.tensor(overlaps, dtype=torch.float32).mean().item()
    )


def _compute_metrics(
    *,
    teacher_costs_cpu: torch.Tensor,
    student_costs_cpu: torch.Tensor,
    candidate_actions_cpu: torch.Tensor,
    teacher_best_index_cpu: torch.Tensor,
    teacher_best_first_action_cpu: torch.Tensor,
) -> dict[str, Any]:
    finite = bool(torch.isfinite(student_costs_cpu).all().item())
    if not finite:
        return {
            "finite_student_costs": False,
            "spearman": float("nan"),
            "top1_overlap": float("nan"),
            "top5_overlap": float("nan"),
            "top10_overlap": float("nan"),
            "regret": float("nan"),
            "first_action_error": float("nan"),
            "teacher_best_index_match_rate": 0.0,
        }

    teacher_sorted = torch.argsort(teacher_costs_cpu, dim=1)
    student_sorted = torch.argsort(student_costs_cpu, dim=1)
    student_best = student_sorted[:, 0]
    teacher_best = teacher_sorted[:, 0]
    regret = (
        teacher_costs_cpu[
            torch.arange(teacher_costs_cpu.shape[0]),
            student_best,
        ]
        - teacher_costs_cpu[
            torch.arange(teacher_costs_cpu.shape[0]),
            teacher_best,
        ]
    )
    student_best_first_action = candidate_actions_cpu[
        torch.arange(candidate_actions_cpu.shape[0]),
        student_best,
        0,
        :,
    ]
    first_action_error = torch.linalg.norm(
        student_best_first_action - teacher_best_first_action_cpu,
        dim=1,
    )
    spearman_vals = torch.tensor(
        [
            _spearman(teacher_costs_cpu[i], student_costs_cpu[i])
            for i in range(teacher_costs_cpu.shape[0])
        ],
        dtype=torch.float32,
    )
    return {
        "finite_student_costs": True,
        "spearman": float(spearman_vals.mean().item()),
        "top1_overlap": _topk_overlap_mean(teacher_sorted, student_sorted, 1),
        "top5_overlap": _topk_overlap_mean(teacher_sorted, student_sorted, 5),
        "top10_overlap": _topk_overlap_mean(
            teacher_sorted,
            student_sorted,
            10,
        ),
        "regret": float(regret.mean().item()),
        "first_action_error": float(first_action_error.mean().item()),
        "teacher_best_index_match_rate": float(
            (student_best == teacher_best_index_cpu).float().mean().item()
        ),
    }


def _resolve_model_path(tag: str, compression_root: Path) -> Path | None:
    distilled = compression_root / tag / "distilled_model.pt"
    compressed = compression_root / tag / "compressed_model.pt"
    if distilled.exists():
        return distilled
    if compressed.exists():
        return compressed
    return None


def _load_split_cache(name: str, path: Path) -> SplitCache:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} cache: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return SplitCache(name=name, path=path, payload=payload)


def _build_info_for_states(
    *,
    env_name: str,
    cache_payload: dict[str, Any],
    device: str,
) -> dict[str, torch.Tensor]:
    return build_info_dict_from_cache(
        env_name=env_name,
        episodes_idx=list(cache_payload["episodes_idx"]),
        start_steps=list(cache_payload["start_steps"]),
        goal_offset_steps=int(cache_payload["goal_offset_steps"]),
        device=device,
    )


def _combo_teacher_targets(
    *,
    teacher_model: torch.nn.Module,
    info_cpu: dict[str, torch.Tensor],
    candidate_actions_cpu: torch.Tensor,
    device: str,
    batch_states: int,
    batch_candidates: int,
    empty_cache_between_batches: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_costs = _compute_costs_chunked(
        model=teacher_model,
        info_cpu=info_cpu,
        candidate_actions_cpu=candidate_actions_cpu,
        device=device,
        batch_states=batch_states,
        batch_candidates=batch_candidates,
        empty_cache_between_batches=empty_cache_between_batches,
    )
    sorted_idx = torch.argsort(teacher_costs, dim=1)
    best_idx = sorted_idx[:, 0]
    best_first = candidate_actions_cpu[
        torch.arange(candidate_actions_cpu.shape[0]),
        best_idx,
        0,
        :,
    ]
    return teacher_costs, best_idx, best_first


def _build_combos(
    *,
    env_name: str,
    train_cache: SplitCache,
    eval_cache: SplitCache,
    teacher_model: torch.nn.Module,
    device: str,
    batch_states: int,
    batch_candidates: int,
    empty_cache_between_batches: bool,
) -> list[EvalCombo]:
    train_info_cpu = _build_info_for_states(
        env_name=env_name,
        cache_payload=train_cache.payload,
        device="cpu",
    )
    eval_info_cpu = _build_info_for_states(
        env_name=env_name,
        cache_payload=eval_cache.payload,
        device="cpu",
    )
    train_info_cpu = maybe_align_action_width(train_info_cpu, teacher_model)
    eval_info_cpu = maybe_align_action_width(eval_info_cpu, teacher_model)

    combos: list[EvalCombo] = []
    # exact same-split rows
    combos.append(
        EvalCombo(
            state_split="train",
            candidate_split="train",
            candidate_mode="exact",
            info_cpu=train_info_cpu,
            candidate_actions_cpu=(
                train_cache.payload["candidate_actions"].float()
            ),
            teacher_costs_cpu=train_cache.payload["teacher_costs"].float(),
            teacher_best_index=(
                train_cache.payload["teacher_best_index"].long()
            ),
            teacher_best_first_action=(
                train_cache.payload["teacher_best_first_action"].float()
            ),
        )
    )
    combos.append(
        EvalCombo(
            state_split="eval",
            candidate_split="eval",
            candidate_mode="exact",
            info_cpu=eval_info_cpu,
            candidate_actions_cpu=(
                eval_cache.payload["candidate_actions"].float()
            ),
            teacher_costs_cpu=eval_cache.payload["teacher_costs"].float(),
            teacher_best_index=eval_cache.payload["teacher_best_index"].long(),
            teacher_best_first_action=(
                eval_cache.payload["teacher_best_first_action"].float()
            ),
        )
    )

    # Cross rows via generated candidates from opposite split
    # sampling seed/distribution.
    action_dim = int(
        ENV_SPECS[env_name].action_dim
        or train_cache.payload["candidate_actions"].shape[-1]
    )
    train_seed = int(train_cache.payload.get("seed", 0))
    eval_seed = int(eval_cache.payload.get("seed", 1))
    train_n = int(train_cache.payload["num_states"])
    eval_n = int(eval_cache.payload["num_states"])
    train_c = int(train_cache.payload["num_candidates"])
    eval_c = int(eval_cache.payload["num_candidates"])
    train_h = int(train_cache.payload["horizon"])
    eval_h = int(eval_cache.payload["horizon"])

    train_states_eval_cands = _sample_candidates(
        num_states=train_n,
        num_candidates=eval_c,
        horizon=eval_h,
        action_dim=action_dim,
        seed=eval_seed,
    )
    t_cost, t_best, t_first = _combo_teacher_targets(
        teacher_model=teacher_model,
        info_cpu=train_info_cpu,
        candidate_actions_cpu=train_states_eval_cands,
        device=device,
        batch_states=batch_states,
        batch_candidates=batch_candidates,
        empty_cache_between_batches=empty_cache_between_batches,
    )
    combos.append(
        EvalCombo(
            state_split="train",
            candidate_split="eval",
            candidate_mode="generated_from_eval_seed",
            info_cpu=train_info_cpu,
            candidate_actions_cpu=train_states_eval_cands,
            teacher_costs_cpu=t_cost,
            teacher_best_index=t_best,
            teacher_best_first_action=t_first,
        )
    )

    eval_states_train_cands = _sample_candidates(
        num_states=eval_n,
        num_candidates=train_c,
        horizon=train_h,
        action_dim=action_dim,
        seed=train_seed,
    )
    e_cost, e_best, e_first = _combo_teacher_targets(
        teacher_model=teacher_model,
        info_cpu=eval_info_cpu,
        candidate_actions_cpu=eval_states_train_cands,
        device=device,
        batch_states=batch_states,
        batch_candidates=batch_candidates,
        empty_cache_between_batches=empty_cache_between_batches,
    )
    combos.append(
        EvalCombo(
            state_split="eval",
            candidate_split="train",
            candidate_mode="generated_from_train_seed",
            info_cpu=eval_info_cpu,
            candidate_actions_cpu=eval_states_train_cands,
            teacher_costs_cpu=e_cost,
            teacher_best_index=e_best,
            teacher_best_first_action=e_first,
        )
    )
    return combos


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "model_tag",
        "state_split",
        "candidate_split",
        "candidate_mode",
        "finite_student_costs",
        "spearman",
        "top1_overlap",
        "top5_overlap",
        "top10_overlap",
        "regret",
        "first_action_error",
        "teacher_best_index_match_rate",
    ]
    lines = [
        "# Crossed Operator Evaluation (TwoRoom)",
        "",
        "|" + "|".join(cols) + "|",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]

    def _fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    for row in rows:
        lines.append("|" + "|".join(_fmt(row.get(c)) for c in cols) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
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
    parser.add_argument(
        "--model-tags",
        nargs="*",
        default=DEFAULT_MODEL_TAGS,
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-candidates", type=int, default=128)
    parser.add_argument(
        "--empty-cache-between-batches",
        action="store_true",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    train_cache = _load_split_cache("train", Path(args.train_cache))
    eval_cache = _load_split_cache("eval", Path(args.eval_cache))

    teacher_family = str(train_cache.payload["model_family"])
    teacher_checkpoint = str(
        train_cache.payload.get(
            "resolved_checkpoint",
            train_cache.payload["checkpoint"],
        )
    )
    loaded = load_cost_model(
        family=teacher_family,
        checkpoint=teacher_checkpoint,
        env_name=args.env,
        device=device,
    )
    teacher = loaded.model.to(device).eval()
    teacher.requires_grad_(False)

    combos = _build_combos(
        env_name=args.env,
        train_cache=train_cache,
        eval_cache=eval_cache,
        teacher_model=teacher,
        device=device,
        batch_states=int(args.batch_states),
        batch_candidates=int(args.batch_candidates),
        empty_cache_between_batches=bool(args.empty_cache_between_batches),
    )

    compression_root = Path("outputs/compression") / args.env
    rows: list[dict[str, Any]] = []
    for tag in args.model_tags:
        model_path = _resolve_model_path(tag, compression_root)
        if model_path is None:
            for combo in combos:
                rows.append(
                    {
                        "model_tag": tag,
                        "model_path": None,
                        "state_split": combo.state_split,
                        "candidate_split": combo.candidate_split,
                        "candidate_mode": combo.candidate_mode,
                        "finite_student_costs": False,
                        "spearman": None,
                        "top1_overlap": None,
                        "top5_overlap": None,
                        "top10_overlap": None,
                        "regret": None,
                        "first_action_error": None,
                        "teacher_best_index_match_rate": None,
                        "notes": "missing_model_artifact",
                    }
                )
            continue

        model = load_model_from_path(model_path, device=device)
        model = model.to(device).eval()
        model.requires_grad_(False)
        for combo in combos:
            info_cpu = maybe_align_action_width(dict(combo.info_cpu), model)
            student_costs = _compute_costs_chunked(
                model=model,
                info_cpu=info_cpu,
                candidate_actions_cpu=combo.candidate_actions_cpu,
                device=device,
                batch_states=int(args.batch_states),
                batch_candidates=int(args.batch_candidates),
                empty_cache_between_batches=bool(
                    args.empty_cache_between_batches
                ),
            )
            metrics = _compute_metrics(
                teacher_costs_cpu=combo.teacher_costs_cpu,
                student_costs_cpu=student_costs,
                candidate_actions_cpu=combo.candidate_actions_cpu,
                teacher_best_index_cpu=combo.teacher_best_index,
                teacher_best_first_action_cpu=combo.teacher_best_first_action,
            )
            rows.append(
                {
                    "model_tag": tag,
                    "model_path": str(model_path),
                    "state_split": combo.state_split,
                    "candidate_split": combo.candidate_split,
                    "candidate_mode": combo.candidate_mode,
                    **metrics,
                    "notes": "",
                }
            )

    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "operator_crossed_eval_tworoom.csv"
    json_path = out_dir / "operator_crossed_eval_tworoom.json"
    md_path = out_dir / "operator_crossed_eval_tworoom.md"

    fields = [
        "model_tag",
        "model_path",
        "state_split",
        "candidate_split",
        "candidate_mode",
        "finite_student_costs",
        "spearman",
        "top1_overlap",
        "top5_overlap",
        "top10_overlap",
        "regret",
        "first_action_error",
        "teacher_best_index_match_rate",
        "notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    save_json(
        json_path,
        {
            "env": args.env,
            "train_cache": str(train_cache.path),
            "eval_cache": str(eval_cache.path),
            "teacher_family": teacher_family,
            "teacher_checkpoint": teacher_checkpoint,
            "rows": rows,
        },
    )
    _write_md(md_path, rows)
    print("Crossed operator evaluation written")
    print(f"  csv:  {csv_path}")
    print(f"  json: {json_path}")
    print(f"  md:   {md_path}")


if __name__ == "__main__":
    main()
