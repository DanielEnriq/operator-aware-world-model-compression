from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from oawc.compression.operator_eval import (
    evaluate_model_on_operator_cache,
    load_operator_cache,
)
from oawc.compression.operator_metrics import resolve_device
from oawc.compression.reports import save_json
from oawc.envs import ENV_SPECS


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
    return (
        torch.rand(
            (num_states, num_candidates, horizon, action_dim),
            generator=gen,
            dtype=torch.float32,
        )
        * 2.0
        - 1.0
    )


def _resolve_model_path(tag: str, root: Path) -> Path | None:
    candidate_tags = [tag]
    if tag.endswith("_split"):
        candidate_tags.append(tag.removesuffix("_split"))
    for t in candidate_tags:
        distilled = root / t / "distilled_model.pt"
        compressed = root / t / "compressed_model.pt"
        if distilled.exists():
            return distilled
        if compressed.exists():
            return compressed
    return None


def _first_divergence(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float,
) -> dict[str, Any] | None:
    diff = (a - b).abs()
    bad = diff > atol
    if not bool(bad.any().item()):
        return None
    ij = torch.nonzero(bad, as_tuple=False)[0]
    i = int(ij[0].item())
    j = int(ij[1].item())
    rel = diff[i, j] / max(1e-12, float(a[i, j].abs().item()))
    return {
        "state_index": i,
        "candidate_index": j,
        "old_student_value": float(a[i, j].item()),
        "crossed_student_value": float(b[i, j].item()),
        "abs_diff": float(diff[i, j].item()),
        "rel_diff": float(rel),
    }


def _assert_exact_consistency(
    *,
    model_path: Path,
    cache_path: Path,
    result_exact: dict[str, Any],
    result_probe: dict[str, Any],
    atol: float,
    label: str,
) -> dict[str, Any]:
    cand_equal = bool(
        torch.equal(
            result_exact["candidate_actions"],
            result_probe["candidate_actions"],
        )
    )
    teacher_equal = bool(
        torch.equal(
            result_exact["teacher_costs"],
            result_probe["teacher_costs"],
        )
    )
    student_allclose = bool(
        torch.allclose(
            result_exact["student_costs"],
            result_probe["student_costs"],
            atol=atol,
            rtol=0.0,
        )
    )
    max_abs = float(
        (result_exact["student_costs"] - result_probe["student_costs"])
        .abs()
        .max()
        .item()
    )
    max_rel = float(
        (
            (result_exact["student_costs"] - result_probe["student_costs"]).abs()
            / result_exact["student_costs"].abs().clamp_min(1e-12)
        )
        .max()
        .item()
    )
    first = _first_divergence(
        result_exact["student_costs"],
        result_probe["student_costs"],
        atol,
    )
    report = {
        "label": label,
        "model_path": str(model_path),
        "cache_path": str(cache_path),
        "candidate_equal": cand_equal,
        "teacher_equal": teacher_equal,
        "student_allclose": student_allclose,
        "student_max_abs_diff": max_abs,
        "student_max_rel_diff": max_rel,
        "first_divergence": first,
    }
    if not (cand_equal and teacher_equal and student_allclose):
        raise RuntimeError(
            f"Exact-mode consistency assertion failed for {label} "
            f"(candidate_equal={cand_equal}, teacher_equal={teacher_equal}, "
            f"student_allclose={student_allclose}, max_abs_diff={max_abs}, "
            f"max_rel_diff={max_rel}, model_path={model_path}, "
            f"cache_path={cache_path}, first_divergence={first})"
        )
    return report


def _row_from_result(
    *,
    model_tag: str,
    model_path: Path,
    state_split: str,
    candidate_split: str,
    candidate_mode: str,
    result: dict[str, Any],
    notes: str,
) -> dict[str, Any]:
    m = result["metrics"]
    return {
        "model_tag": model_tag,
        "model_path": str(model_path),
        "state_split": state_split,
        "candidate_split": candidate_split,
        "candidate_mode": candidate_mode,
        "finite_student_costs": m["finite_student_costs"],
        "spearman": m["spearman_per_state"]["mean"],
        "top1_overlap": m["topk_overlap"].get("1", {}).get("mean"),
        "top5_overlap": m["topk_overlap"].get("5", {}).get("mean"),
        "top10_overlap": m["topk_overlap"].get("10", {}).get("mean"),
        "regret": m["teacher_regret"]["mean"],
        "first_action_error": m["selected_first_action_error"]["mean"],
        "teacher_best_index_match_rate": m["teacher_best_index_match_rate"],
        "notes": notes,
    }


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
        "notes",
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
    parser.add_argument("--model-tags", nargs="*", default=DEFAULT_MODEL_TAGS)
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-candidates", type=int, default=128)
    parser.add_argument("--debug-assert-exact-consistency", action="store_true")
    parser.add_argument("--debug-atol", type=float, default=1e-3)
    args = parser.parse_args()

    device = resolve_device(args.device)
    train_cache = load_operator_cache(args.train_cache)
    eval_cache = load_operator_cache(args.eval_cache)
    root = Path("outputs/compression") / args.env

    action_dim = int(
        ENV_SPECS[args.env].action_dim or train_cache.candidate_actions.shape[-1]
    )
    train_seed = int(train_cache.payload.get("seed", 0))
    eval_seed = int(eval_cache.payload.get("seed", 1))

    rows: list[dict[str, Any]] = []
    debug_checks: list[dict[str, Any]] = []

    for tag in args.model_tags:
        model_path = _resolve_model_path(tag, root)
        if model_path is None:
            for state_split, candidate_split, candidate_mode in [
                ("train", "train", "exact"),
                ("eval", "eval", "exact"),
                ("train", "eval", "generated"),
                ("eval", "train", "generated"),
            ]:
                rows.append(
                    {
                        "model_tag": tag,
                        "model_path": None,
                        "state_split": state_split,
                        "candidate_split": candidate_split,
                        "candidate_mode": candidate_mode,
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

        # Exact train/train
        res_tt = evaluate_model_on_operator_cache(
            cache=train_cache,
            model_path=model_path,
            device=device,
            use_chunked_student=False,
        )
        rows.append(
            _row_from_result(
                model_tag=tag,
                model_path=model_path,
                state_split="train",
                candidate_split="train",
                candidate_mode="exact",
                result=res_tt,
                notes="",
            )
        )

        # Exact eval/eval
        res_ee = evaluate_model_on_operator_cache(
            cache=eval_cache,
            model_path=model_path,
            device=device,
            use_chunked_student=False,
        )
        rows.append(
            _row_from_result(
                model_tag=tag,
                model_path=model_path,
                state_split="eval",
                candidate_split="eval",
                candidate_mode="exact",
                result=res_ee,
                notes="",
            )
        )

        if args.debug_assert_exact_consistency:
            probe_tt = evaluate_model_on_operator_cache(
                cache=train_cache,
                model_path=model_path,
                device=device,
                use_chunked_student=True,
                batch_states=int(args.batch_states),
                batch_candidates=int(args.batch_candidates),
            )
            debug_checks.append(
                _assert_exact_consistency(
                    model_path=model_path,
                    cache_path=train_cache.path,
                    result_exact=res_tt,
                    result_probe=probe_tt,
                    atol=float(args.debug_atol),
                    label=f"{tag} [train/train]",
                )
            )
            probe_ee = evaluate_model_on_operator_cache(
                cache=eval_cache,
                model_path=model_path,
                device=device,
                use_chunked_student=True,
                batch_states=int(args.batch_states),
                batch_candidates=int(args.batch_candidates),
            )
            debug_checks.append(
                _assert_exact_consistency(
                    model_path=model_path,
                    cache_path=eval_cache.path,
                    result_exact=res_ee,
                    result_probe=probe_ee,
                    atol=float(args.debug_atol),
                    label=f"{tag} [eval/eval]",
                )
            )

        # Generated train/eval candidates, teacher recomputed
        gen_te = _sample_candidates(
            num_states=int(train_cache.candidate_actions.shape[0]),
            num_candidates=int(eval_cache.candidate_actions.shape[1]),
            horizon=int(eval_cache.payload["horizon"]),
            action_dim=action_dim,
            seed=eval_seed,
        )
        res_te = evaluate_model_on_operator_cache(
            cache=train_cache,
            model_path=model_path,
            device=device,
            candidate_actions_override=gen_te,
            batch_states=int(args.batch_states),
            batch_candidates=int(args.batch_candidates),
            use_chunked_student=True,
        )
        rows.append(
            _row_from_result(
                model_tag=tag,
                model_path=model_path,
                state_split="train",
                candidate_split="eval",
                candidate_mode="generated",
                result=res_te,
                notes=f"teacher_source={res_te['metadata']['teacher_source']}",
            )
        )

        # Generated eval/train candidates, teacher recomputed
        gen_et = _sample_candidates(
            num_states=int(eval_cache.candidate_actions.shape[0]),
            num_candidates=int(train_cache.candidate_actions.shape[1]),
            horizon=int(train_cache.payload["horizon"]),
            action_dim=action_dim,
            seed=train_seed,
        )
        res_et = evaluate_model_on_operator_cache(
            cache=eval_cache,
            model_path=model_path,
            device=device,
            candidate_actions_override=gen_et,
            batch_states=int(args.batch_states),
            batch_candidates=int(args.batch_candidates),
            use_chunked_student=True,
        )
        rows.append(
            _row_from_result(
                model_tag=tag,
                model_path=model_path,
                state_split="eval",
                candidate_split="train",
                candidate_mode="generated",
                result=res_et,
                notes=f"teacher_source={res_et['metadata']['teacher_source']}",
            )
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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    save_json(
        json_path,
        {
            "env": args.env,
            "train_cache": str(train_cache.path),
            "eval_cache": str(eval_cache.path),
            "rows": rows,
            "debug_exact_consistency": debug_checks,
        },
    )
    _write_md(md_path, rows)
    print("Crossed operator evaluation written")
    print(f"  csv:  {csv_path}")
    print(f"  json: {json_path}")
    print(f"  md:   {md_path}")


if __name__ == "__main__":
    main()
