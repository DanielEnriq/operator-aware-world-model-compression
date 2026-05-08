from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from oawc.compression.operator_eval import rank_fraction_to_tag

METHOD_ORDER = {
    "teacher": 0,
    "weight_svd": 1,
    "activation_svd": 2,
    "operator_cost_kl": 3,
    "operator_hybrid": 4,
    "operator_elite": 5,
}
OPERATOR_METHOD_TO_SCRIPT = {
    "operator_cost_kl": "scripts/distill_operator_cost_kl.py",
    "operator_hybrid": "scripts/distill_operator_hybrid.py",
    "operator_elite": "scripts/distill_operator_elite.py",
}
TABLE_COLUMNS = [
    "env",
    "model_family",
    "teacher_checkpoint",
    "base_checkpoint_or_path",
    "method",
    "init_method",
    "rank_fraction",
    "predictor_compression_ratio",
    "total_compression_ratio",
    "original_total_params",
    "compressed_total_params",
    "original_predictor_params",
    "compressed_predictor_params",
    "layers_compressed",
    "compression_status",
    "distill_steps",
    "distill_batch_size",
    "distill_train_cache",
    "distill_loss",
    "distill_wall_time_s",
    "num_eval",
    "seed",
    "success_rate",
    "num_successes",
    "avg_return",
    "avg_final_distance",
    "eval_time_s",
    "model_path",
    "benchmark_json",
    "compression_report",
    "distill_report",
    "status",
    "error_message",
    "notes",
]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_str(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _normalize_output_base(value: str) -> Path:
    raw = Path(value)
    if raw.name.endswith(".*"):
        return raw.with_name(raw.name[:-2])
    if raw.suffix in {".csv", ".json", ".md"}:
        return raw.with_suffix("")
    return raw


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_md(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _family_short_tag(model_family: str) -> str:
    if model_family == "lewm_hf":
        return "lewm"
    if model_family == "swm_prejepa_local":
        return "swm_prejepa"
    return model_family


def _method_prefix(env_name: str, model_family: str, method: str) -> str:
    family = _family_short_tag(model_family)
    return f"{family}_{env_name}_{method}"


def _benchmark_json_path(
    env_name: str,
    tag: str,
    seed: int,
    num_eval: int,
) -> Path:
    return (
        Path("outputs/benchmarks")
        / env_name
        / tag
        / f"{tag}_seed{seed}_n{num_eval}.json"
    )


def _run(
    cmd: list[str],
    env: dict[str, str],
    *,
    dry_run: bool,
) -> tuple[bool, float, str]:
    print("[cmd]", " ".join(cmd))
    if dry_run:
        return True, 0.0, "[dry-run] command not executed"
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    code = proc.wait()
    return code == 0, (time.time() - start), "".join(lines)[-6000:]


def _benchmark_teacher_cmd(
    *,
    env_name: str,
    model_family: str,
    checkpoint: str,
    tag: str,
    num_eval: int,
    seed: int,
    device: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/benchmark_cost_model.py",
        "--env",
        env_name,
        "--model-family",
        model_family,
        "--checkpoint",
        checkpoint,
        "--tag",
        tag,
        "--num-eval",
        str(num_eval),
        "--seed",
        str(seed),
        "--device",
        device,
    ]


def _benchmark_local_cmd(
    *,
    env_name: str,
    model_path: Path,
    tag: str,
    num_eval: int,
    seed: int,
    device: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/benchmark_cost_model.py",
        "--env",
        env_name,
        "--model-path",
        str(model_path),
        "--tag",
        tag,
        "--num-eval",
        str(num_eval),
        "--seed",
        str(seed),
        "--device",
        device,
    ]


def _build_cache_cmd(
    *,
    env_name: str,
    model_family: str,
    checkpoint: str,
    cache_tag: str,
    num_states: int,
    num_candidates: int,
    horizon: int,
    seed: int,
    device: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/build_operator_cache.py",
        "--env",
        env_name,
        "--model-family",
        model_family,
        "--checkpoint",
        checkpoint,
        "--num-states",
        str(num_states),
        "--num-candidates",
        str(num_candidates),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
        "--device",
        device,
        "--tag",
        cache_tag,
        "--split",
        "train",
    ]


def _compression_cmd(
    *,
    method: str,
    env_name: str,
    model_family: str,
    teacher_checkpoint: str,
    rank_fraction: float,
    target_substrings: str,
    device: str,
    tag: str,
) -> list[str]:
    if method == "weight_svd":
        return [
            "uv",
            "run",
            "python",
            "scripts/compress_predictor_weight_svd.py",
            "--env",
            env_name,
            "--model-family",
            model_family,
            "--checkpoint",
            teacher_checkpoint,
            "--rank-fraction",
            str(rank_fraction),
            "--target-substrings",
            target_substrings,
            "--min-dim",
            "64",
            "--device",
            device,
            "--tag",
            tag,
        ]
    return [
        "uv",
        "run",
        "python",
        "scripts/compress_predictor_activation_svd.py",
        "--env",
        env_name,
        "--model-family",
        model_family,
        "--checkpoint",
        teacher_checkpoint,
        "--rank-fraction",
        str(rank_fraction),
        "--target-substrings",
        target_substrings,
        "--min-dim",
        "64",
        "--num-calib-states",
        "64",
        "--num-calib-candidates",
        "64",
        "--horizon",
        "5",
        "--max-rows-per-layer",
        "8192",
        "--ridge",
        "1e-4",
        "--calib-batch-states",
        "8",
        "--calib-batch-candidates",
        "128",
        "--device",
        device,
        "--tag",
        tag,
    ]


def _distill_cmd(
    *,
    method: str,
    env_name: str,
    train_cache: Path,
    student_path: Path,
    tag: str,
    device: str,
    seed: int,
    distill_steps: int,
    distill_batch_states: int,
    distill_batch_candidates: int,
    distill_lr: float,
    distill_weight_decay: float,
    eval_every: int,
    val_states: int,
    val_candidates: int,
    save_best_by_val: bool,
) -> list[str]:
    script = OPERATOR_METHOD_TO_SCRIPT[method]
    cmd = [
        "uv",
        "run",
        "python",
        script,
        "--env",
        env_name,
        "--train-cache",
        str(train_cache),
        "--student-path",
        str(student_path),
        "--max-steps",
        str(distill_steps),
        "--batch-states",
        str(distill_batch_states),
        "--batch-candidates",
        str(distill_batch_candidates),
        "--lr",
        str(distill_lr),
        "--weight-decay",
        str(distill_weight_decay),
        "--eval-every",
        str(eval_every),
        "--val-states",
        str(val_states),
        "--val-candidates",
        str(val_candidates),
        "--seed",
        str(seed),
        "--device",
        device,
        "--tag",
        tag,
    ]
    if save_best_by_val:
        cmd.append("--save-best-by-val")
    return cmd


def _extract_metric(
    raw_metrics: dict[str, Any],
    keys: list[str],
) -> float | None:
    for key in keys:
        if key not in raw_metrics:
            continue
        value = raw_metrics[key]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for field in ("mean", "avg", "value"):
                field_value = value.get(field)
                if isinstance(field_value, (int, float)):
                    return float(field_value)
    return None


def _populate_benchmark_metrics(
    row: dict[str, Any],
    bench: dict[str, Any],
) -> None:
    performance = bench.get("performance", {})
    efficiency = bench.get("efficiency", {})
    row["success_rate"] = performance.get("success_rate")
    row["num_successes"] = performance.get("num_successes")
    if row["num_successes"] is None:
        episode_successes = performance.get("episode_successes")
        if isinstance(episode_successes, list):
            row["num_successes"] = int(
                sum(int(bool(x)) for x in episode_successes)
            )
        elif isinstance(episode_successes, (int, float)):
            row["num_successes"] = int(episode_successes)
    row["eval_time_s"] = efficiency.get("evaluation_time_sec")
    row["avg_return"] = performance.get("avg_return")
    row["avg_final_distance"] = performance.get("avg_final_distance")
    raw_metrics = performance.get("raw_metrics", {})
    if row["avg_return"] is None:
        row["avg_return"] = _extract_metric(
            raw_metrics,
            ["avg_return", "mean_return", "return"],
        )
    if row["avg_final_distance"] is None:
        row["avg_final_distance"] = _extract_metric(
            raw_metrics,
            ["avg_final_distance", "final_distance", "goal_distance"],
        )


def _apply_compression_report(
    row: dict[str, Any],
    report_path: Path,
) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    report = _read_json(report_path)
    row["compression_report"] = str(report_path)
    row["predictor_compression_ratio"] = report.get("predictor_compression_ratio")
    row["total_compression_ratio"] = report.get("total_compression_ratio")
    row["original_total_params"] = report.get("total_params_before")
    row["compressed_total_params"] = report.get("total_params_after")
    row["original_predictor_params"] = report.get("predictor_params_before")
    row["compressed_predictor_params"] = report.get("predictor_params_after")
    row["layers_compressed"] = report.get(
        "layers_compressed",
        report.get("num_layers_compressed"),
    )
    row["compression_status"] = report.get("compression_status")
    return report


def _apply_distill_report(
    row: dict[str, Any],
    report_path: Path,
) -> dict[str, Any] | None:
    if not report_path.exists():
        return None
    report = _read_json(report_path)
    row["distill_report"] = str(report_path)
    row["distill_steps"] = report.get("distill_steps", report.get("max_steps"))
    row["distill_batch_size"] = report.get(
        "batch_size",
        report.get("distill_batch_states"),
    )
    row["distill_train_cache"] = report.get("train_cache")
    row["distill_loss"] = report.get(
        "final_training_loss",
        report.get("final_total_loss", report.get("final_train_kl")),
    )
    row["distill_wall_time_s"] = report.get("wall_time_sec")
    return report


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["env"]),
            str(row["model_family"]),
            METHOD_ORDER.get(str(row["method"]), 99),
            -float(row["rank_fraction"])
            if isinstance(row.get("rank_fraction"), (int, float))
            else 0.0,
        ),
    )


def _write_table_bundle(
    *,
    rows: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
    md_path: Path,
    title: str,
) -> None:
    rows = _sort_rows(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in TABLE_COLUMNS})

    payload = {
        "schema_version": 1,
        "kind": "full_closed_loop_frontier",
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {title}",
        "",
        "Closed-loop success from `scripts/benchmark_cost_model.py` is the primary metric.",
        "Operator caches are used only for operator-aware distillation training.",
        "",
        "|" + "|".join(TABLE_COLUMNS) + "|",
        "|" + "|".join(["---"] * len(TABLE_COLUMNS)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(_fmt_md(row.get(c)) for c in TABLE_COLUMNS) + "|")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _new_row(
    *,
    env_name: str,
    model_family: str,
    teacher_checkpoint: str,
    method: str,
    init_method: str,
    rank_fraction: float | None,
    num_eval: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "env": env_name,
        "model_family": model_family,
        "teacher_checkpoint": teacher_checkpoint,
        "base_checkpoint_or_path": teacher_checkpoint,
        "method": method,
        "init_method": init_method,
        "rank_fraction": rank_fraction,
        "predictor_compression_ratio": None,
        "total_compression_ratio": None,
        "original_total_params": None,
        "compressed_total_params": None,
        "original_predictor_params": None,
        "compressed_predictor_params": None,
        "layers_compressed": None,
        "compression_status": None,
        "distill_steps": None,
        "distill_batch_size": None,
        "distill_train_cache": None,
        "distill_loss": None,
        "distill_wall_time_s": None,
        "num_eval": int(num_eval),
        "seed": int(seed),
        "success_rate": None,
        "num_successes": None,
        "avg_return": None,
        "avg_final_distance": None,
        "eval_time_s": None,
        "model_path": None,
        "benchmark_json": None,
        "compression_report": None,
        "distill_report": None,
        "status": "pending",
        "error_message": None,
        "notes": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument(
        "--model-family",
        required=True,
        choices=["lewm_hf", "swm_prejepa_local"],
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--ranks", default="0.90,0.80,0.70,0.60,0.50")
    parser.add_argument(
        "--methods",
        default=(
            "weight_svd,activation_svd,operator_cost_kl,operator_hybrid"
        ),
    )
    parser.add_argument(
        "--operator-init",
        default="weight_svd",
        choices=["weight_svd", "activation_svd"],
    )
    parser.add_argument("--num-eval", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compress-if-missing", action="store_true")
    parser.add_argument("--distill-if-missing", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--target-substrings", default="predictor")
    parser.add_argument("--train-operator-cache", default=None)
    parser.add_argument("--build-train-cache-if-missing", action="store_true")
    parser.add_argument("--num-train-states", type=int, default=512)
    parser.add_argument("--num-train-candidates", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--distill-steps", type=int, default=1000)
    parser.add_argument("--distill-batch-states", type=int, default=16)
    parser.add_argument("--distill-batch-candidates", type=int, default=64)
    parser.add_argument("--distill-lr", type=float, default=1e-4)
    parser.add_argument("--distill-weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--val-states", type=int, default=64)
    parser.add_argument("--val-candidates", type=int, default=128)
    parser.add_argument("--save-best-by-val", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/tables/full_closed_loop_frontier_all.*",
    )
    args = parser.parse_args()

    env_name = args.env
    methods = _parse_csv_str(args.methods)
    ranks = _parse_csv_floats(args.ranks)
    output_base = _normalize_output_base(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    env_vars = os.environ.copy()
    env_vars.setdefault("PYTHONPATH", str(Path.cwd() / "src"))
    family_prefix = _family_short_tag(args.model_family)
    default_cache_tag = (
        f"{family_prefix}_{env_name}_train_s{args.num_train_states}_"
        f"c{args.num_train_candidates}_seed{args.seed}"
    )
    train_cache_path = (
        Path(args.train_operator_cache)
        if args.train_operator_cache
        else (
            Path("outputs/operator_cache")
            / env_name
            / default_cache_tag
            / "operator_cache.pt"
        )
    )

    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "full_closed_loop_frontier_run",
        "env": env_name,
        "model_family": args.model_family,
        "teacher_checkpoint": args.teacher_checkpoint,
        "methods": methods,
        "ranks": ranks,
        "seed": int(args.seed),
        "num_eval": int(args.num_eval),
        "commands": [],
        "rows": [],
    }

    teacher_tag = (
        f"{family_prefix}_{env_name}_teacher_closed_loop_seed{args.seed}_n{args.num_eval}"
    )
    teacher_row = _new_row(
        env_name=env_name,
        model_family=args.model_family,
        teacher_checkpoint=args.teacher_checkpoint,
        method="teacher",
        init_method="none",
        rank_fraction=1.0,
        num_eval=args.num_eval,
        seed=args.seed,
    )
    teacher_row["notes"] = "teacher_anchor"
    teacher_row["compression_status"] = "teacher_anchor"
    teacher_bench = _benchmark_json_path(env_name, teacher_tag, args.seed, args.num_eval)
    teacher_row["benchmark_json"] = str(teacher_bench)
    if args.skip_existing and teacher_bench.exists():
        teacher_row["status"] = "skipped_existing"
    else:
        cmd = _benchmark_teacher_cmd(
            env_name=env_name,
            model_family=args.model_family,
            checkpoint=args.teacher_checkpoint,
            tag=teacher_tag,
            num_eval=args.num_eval,
            seed=args.seed,
            device=args.device,
        )
        manifest["commands"].append(cmd)
        ok, _elapsed, tail = _run(cmd, env_vars, dry_run=args.dry_run)
        if not ok:
            teacher_row["status"] = "failed"
            teacher_row["error_message"] = tail
            rows.append(teacher_row)
            if args.fail_fast:
                raise RuntimeError(tail)
        else:
            if args.dry_run:
                teacher_row["status"] = "dry_run"
            elif teacher_bench.exists():
                bench = _read_json(teacher_bench)
                _populate_benchmark_metrics(teacher_row, bench)
                teacher_row["status"] = "ok"
            else:
                teacher_row["status"] = "failed"
                teacher_row["error_message"] = f"benchmark output missing: {teacher_bench}"
    rows.append(teacher_row)
    manifest["rows"].append(dict(teacher_row))

    cache_required = any(m in OPERATOR_METHOD_TO_SCRIPT for m in methods)
    if cache_required and not train_cache_path.exists():
        if args.build_train_cache_if_missing:
            cache_tag = train_cache_path.parent.name
            cmd = _build_cache_cmd(
                env_name=env_name,
                model_family=args.model_family,
                checkpoint=args.teacher_checkpoint,
                cache_tag=cache_tag,
                num_states=args.num_train_states,
                num_candidates=args.num_train_candidates,
                horizon=args.horizon,
                seed=args.seed,
                device=args.device,
            )
            manifest["commands"].append(cmd)
            ok, _elapsed, tail = _run(cmd, env_vars, dry_run=args.dry_run)
            if not ok:
                if args.fail_fast:
                    raise RuntimeError(tail)
        elif not args.dry_run:
            raise FileNotFoundError(
                "train operator cache missing; provide --train-operator-cache "
                "or --build-train-cache-if-missing."
            )

    for method in methods:
        if method not in METHOD_ORDER:
            raise ValueError(f"Unsupported method: {method}")
        if method == "teacher":
            continue
        for rank in ranks:
            rank_tag = rank_fraction_to_tag(rank)
            row = _new_row(
                env_name=env_name,
                model_family=args.model_family,
                teacher_checkpoint=args.teacher_checkpoint,
                method=method,
                init_method="none" if method in {"weight_svd", "activation_svd"} else args.operator_init,
                rank_fraction=float(rank),
                num_eval=args.num_eval,
                seed=args.seed,
            )
            row["notes"] = "closed_loop_primary_metric; operator_cache_train_only"
            compression_method = method
            if method in OPERATOR_METHOD_TO_SCRIPT:
                compression_method = args.operator_init
                row["init_method"] = args.operator_init

            comp_tag = f"{_method_prefix(env_name, args.model_family, compression_method)}_{rank_tag}"
            comp_dir = Path("outputs/compression") / env_name / comp_tag
            comp_model = comp_dir / "compressed_model.pt"
            comp_report = comp_dir / "compression_report.json"
            row["compression_report"] = str(comp_report)
            row["base_checkpoint_or_path"] = str(comp_model)

            if not comp_model.exists():
                if args.compress_if_missing:
                    cmd = _compression_cmd(
                        method=compression_method,
                        env_name=env_name,
                        model_family=args.model_family,
                        teacher_checkpoint=args.teacher_checkpoint,
                        rank_fraction=float(rank),
                        target_substrings=args.target_substrings,
                        device=args.device,
                        tag=comp_tag,
                    )
                    manifest["commands"].append(cmd)
                    ok, _elapsed, tail = _run(cmd, env_vars, dry_run=args.dry_run)
                    if not ok:
                        row["status"] = "failed"
                        row["compression_status"] = "failed"
                        row["error_message"] = tail
                        rows.append(row)
                        manifest["rows"].append(dict(row))
                        if args.fail_fast:
                            raise RuntimeError(tail)
                        continue
                else:
                    row["status"] = "failed"
                    row["compression_status"] = "missing_compression_artifact"
                    row["error_message"] = (
                        "compression artifact missing; rerun with --compress-if-missing"
                    )
                    rows.append(row)
                    manifest["rows"].append(dict(row))
                    if args.fail_fast:
                        raise RuntimeError(row["error_message"])
                    continue

            comp_payload = None
            if comp_report.exists():
                comp_payload = _apply_compression_report(row, comp_report)
            if (
                comp_payload is not None
                and comp_payload.get("compression_status") == "no_op"
            ):
                row["status"] = "skipped_no_op"
                row["notes"] = (
                    f"{row['notes']}; no-op compression point retained for transparency"
                )
                rows.append(row)
                manifest["rows"].append(dict(row))
                continue

            eval_tag = (
                f"{comp_tag}_closed_loop_seed{args.seed}_n{args.num_eval}"
                if method not in OPERATOR_METHOD_TO_SCRIPT
                else (
                    f"{_method_prefix(env_name, args.model_family, method)}"
                    f"_init_{args.operator_init}_{rank_tag}"
                    f"_closed_loop_seed{args.seed}_n{args.num_eval}"
                )
            )

            model_path = comp_model
            distill_report = None
            if method in OPERATOR_METHOD_TO_SCRIPT:
                distill_tag = (
                    f"{_method_prefix(env_name, args.model_family, method)}"
                    f"_init_{args.operator_init}_{rank_tag}"
                )
                distill_dir = Path("outputs/compression") / env_name / distill_tag
                distill_model = distill_dir / "distilled_model.pt"
                distill_report = distill_dir / "distillation_report.json"
                row["distill_report"] = str(distill_report)
                row["distill_train_cache"] = str(train_cache_path)
                if not distill_model.exists():
                    if args.distill_if_missing:
                        cmd = _distill_cmd(
                            method=method,
                            env_name=env_name,
                            train_cache=train_cache_path,
                            student_path=comp_model,
                            tag=distill_tag,
                            device=args.device,
                            seed=args.seed,
                            distill_steps=args.distill_steps,
                            distill_batch_states=args.distill_batch_states,
                            distill_batch_candidates=args.distill_batch_candidates,
                            distill_lr=args.distill_lr,
                            distill_weight_decay=args.distill_weight_decay,
                            eval_every=args.eval_every,
                            val_states=args.val_states,
                            val_candidates=args.val_candidates,
                            save_best_by_val=args.save_best_by_val,
                        )
                        manifest["commands"].append(cmd)
                        ok, _elapsed, tail = _run(cmd, env_vars, dry_run=args.dry_run)
                        if not ok:
                            row["status"] = "failed"
                            row["error_message"] = tail
                            rows.append(row)
                            manifest["rows"].append(dict(row))
                            if args.fail_fast:
                                raise RuntimeError(tail)
                            continue
                    else:
                        row["status"] = "failed"
                        row["error_message"] = (
                            "distillation artifact missing; rerun with --distill-if-missing"
                        )
                        rows.append(row)
                        manifest["rows"].append(dict(row))
                        if args.fail_fast:
                            raise RuntimeError(row["error_message"])
                        continue
                model_path = distill_model
                row["base_checkpoint_or_path"] = str(comp_model)
                row["model_path"] = str(model_path)
                if distill_report is not None and distill_report.exists():
                    _apply_distill_report(row, distill_report)
            else:
                row["model_path"] = str(model_path)

            bench_path = _benchmark_json_path(
                env_name,
                eval_tag,
                args.seed,
                args.num_eval,
            )
            row["benchmark_json"] = str(bench_path)
            if args.skip_existing and bench_path.exists():
                row["status"] = "skipped_existing"
            else:
                cmd = _benchmark_local_cmd(
                    env_name=env_name,
                    model_path=model_path,
                    tag=eval_tag,
                    num_eval=args.num_eval,
                    seed=args.seed,
                    device=args.device,
                )
                manifest["commands"].append(cmd)
                ok, _elapsed, tail = _run(cmd, env_vars, dry_run=args.dry_run)
                if not ok:
                    row["status"] = "failed"
                    row["error_message"] = tail
                    rows.append(row)
                    manifest["rows"].append(dict(row))
                    if args.fail_fast:
                        raise RuntimeError(tail)
                    continue

            if args.dry_run:
                row["status"] = "dry_run"
            elif bench_path.exists():
                bench = _read_json(bench_path)
                _populate_benchmark_metrics(row, bench)
                row["status"] = (
                    "ok" if row["status"] == "pending" else row["status"]
                )
            else:
                row["status"] = "failed"
                row["error_message"] = f"benchmark output missing: {bench_path}"
                if args.fail_fast:
                    raise RuntimeError(row["error_message"])

            rows.append(row)
            manifest["rows"].append(dict(row))

    run_manifest_path = (
        Path("outputs/tables")
        / f"full_closed_loop_run_manifest_{env_name}_{args.model_family}.json"
    )
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    combined_csv = output_base.with_suffix(".csv")
    combined_json = output_base.with_suffix(".json")
    combined_md = output_base.with_suffix(".md")
    _write_table_bundle(
        rows=rows,
        csv_path=combined_csv,
        json_path=combined_json,
        md_path=combined_md,
        title=(
            f"Full Closed-Loop Frontier ({env_name}, {args.model_family})"
        ),
    )
    print(f"[done] wrote {combined_csv}")
    print(f"[done] wrote {combined_json}")
    print(f"[done] wrote {combined_md}")
    print(f"[done] wrote {run_manifest_path}")


if __name__ == "__main__":
    main()
