from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from oawc.benchmark import load_hdf5_dataset
from oawc.compression.operator_eval import rank_fraction_to_tag

SUPPORTED_ENVS = ("tworoom", "pusht", "ogbench_cube")
DEFAULT_TEACHER_CHECKPOINTS = {
    "tworoom": "quentinll/lewm-tworooms",
    "pusht": "quentinll/lewm-pusht",
    "ogbench_cube": "quentinll/lewm-cube",
}
METHOD_ORDER = {"teacher": 0, "weight_svd": 1, "aa_svd": 2}
TABLE_COLUMNS = [
    "env",
    "tag",
    "model_type",
    "method",
    "rank_fraction",
    "predictor_compression_ratio",
    "total_compression_ratio",
    "layers_compressed",
    "compression_status",
    "num_eval",
    "seed",
    "success_rate",
    "num_successes",
    "eval_time_s",
    "avg_return",
    "avg_final_distance",
    "model_source",
    "checkpoint",
    "model_path",
    "interface_call_path",
    "status",
    "error_message",
]
WARNING_TEXT = (
    "This table reports direct closed-loop MPC benchmark results only. "
    "Random-cache/operator-cache metrics are not used because the "
    "no-compression local artifact identity invariant failed in the earlier "
    "pipeline."
)


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


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


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
            numeric_values = [
                float(v)
                for v in value.values()
                if isinstance(v, (int, float))
            ]
            maybe_mean = _mean_or_none(numeric_values)
            if maybe_mean is not None:
                return maybe_mean
        if isinstance(value, list):
            numeric_values = [
                float(v) for v in value if isinstance(v, (int, float))
            ]
            maybe_mean = _mean_or_none(numeric_values)
            if maybe_mean is not None:
                return maybe_mean
    return None


def _parse_envs(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    for env_name in args.env:
        requested.append(env_name)
    if args.envs:
        requested.extend(_parse_csv_str(args.envs))
    if not requested:
        requested = list(SUPPORTED_ENVS)
    deduped: list[str] = []
    for env_name in requested:
        if env_name not in SUPPORTED_ENVS:
            valid_envs = ", ".join(SUPPORTED_ENVS)
            raise ValueError(
                f"Unsupported env '{env_name}'. Valid envs: {valid_envs}"
            )
        if env_name not in deduped:
            deduped.append(env_name)
    return deduped


def _teacher_checkpoint_for_env(
    env_name: str,
    args: argparse.Namespace,
) -> str:
    if args.teacher_checkpoint is not None:
        return args.teacher_checkpoint
    if env_name == "tworoom":
        return args.tworoom_checkpoint
    if env_name == "pusht":
        return args.pusht_checkpoint
    if env_name == "ogbench_cube":
        return args.cube_checkpoint
    raise KeyError(f"Unsupported env '{env_name}'")


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
    return code == 0, (time.time() - start), "".join(lines)[-4000:]


def _teacher_tag(env_name: str, seed: int, num_eval: int) -> str:
    return f"lewm_{env_name}_teacher_closed_loop_seed{seed}_n{num_eval}"


def _method_prefix(env_name: str, method: str) -> str:
    if method == "weight_svd":
        return f"lewm_{env_name}_svd"
    if method == "aa_svd":
        return f"lewm_{env_name}_aa_svd"
    raise ValueError(f"Unknown method: {method}")


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


def _compression_cmd(
    *,
    method: str,
    env_name: str,
    teacher_checkpoint: str,
    rank_fraction: float,
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
            "lewm_hf",
            "--checkpoint",
            teacher_checkpoint,
            "--rank-fraction",
            str(rank_fraction),
            "--target-substring",
            "predictor",
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
        "lewm_hf",
        "--checkpoint",
        teacher_checkpoint,
        "--rank-fraction",
        str(rank_fraction),
        "--target-substring",
        "predictor",
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


def _benchmark_teacher_cmd(
    *,
    env_name: str,
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
        "lewm_hf",
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


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank_key(row: dict[str, Any]) -> float:
        rank = row.get("rank_fraction")
        if isinstance(rank, (int, float)):
            return -float(rank)
        return 0.0

    return sorted(
        rows,
        key=lambda row: (
            str(row.get("env", "")),
            0 if row.get("model_type") == "teacher" else 1,
            METHOD_ORDER.get(str(row.get("method", "")), 99),
            rank_key(row),
        ),
    )


def _summarize_env(rows: list[dict[str, Any]], env_name: str) -> list[str]:
    env_rows = [row for row in rows if row["env"] == env_name]
    teacher_rows = [
        row
        for row in env_rows
        if row["model_type"] == "teacher" and row["status"] == "ok"
    ]
    teacher_success = (
        teacher_rows[0].get("success_rate") if teacher_rows else None
    )
    compressed_ok = [
        row
        for row in env_rows
        if row["model_type"] == "compressed"
        and row["status"] == "ok"
        and isinstance(row.get("success_rate"), (int, float))
    ]
    best_compressed = (
        max(float(row["success_rate"]) for row in compressed_ok)
        if compressed_ok
        else None
    )
    most_compressed = None
    if compressed_ok:
        with_ratio = [
            row
            for row in compressed_ok
            if isinstance(row.get("predictor_compression_ratio"), (int, float))
        ]
        if with_ratio:
            most_compressed = min(
                with_ratio,
                key=lambda row: float(row["predictor_compression_ratio"]),
            )
    failed_count = len(
        [row for row in env_rows if str(row.get("status")) == "failed"]
    )
    lines = [
        f"- `{env_name}` teacher success_rate: {_fmt_md(teacher_success)}",
        (
            f"- `{env_name}` best compressed success_rate: "
            f"{_fmt_md(best_compressed)}"
        ),
        (
            f"- `{env_name}` highest-compression successful row: "
            f"{most_compressed['tag']} (predictor_ratio="
            f"{_fmt_md(most_compressed.get('predictor_compression_ratio'))}, "
            f"success_rate={_fmt_md(most_compressed.get('success_rate'))})"
            if most_compressed is not None
            else f"- `{env_name}` highest-compression successful row: "
            "none"
        ),
        f"- `{env_name}` failed row count: {failed_count}",
    ]
    return lines


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

    payload = {"rows": rows}
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    envs = sorted({str(row["env"]) for row in rows})
    lines = [
        f"# {title}",
        "",
        f"> **WARNING:** {WARNING_TEXT}",
        "",
        "## Per-env summary",
    ]
    for env_name in envs:
        lines.extend(_summarize_env(rows, env_name))
    lines.extend(
        [
            "",
            "|" + "|".join(TABLE_COLUMNS) + "|",
            "|" + "|".join(["---"] * len(TABLE_COLUMNS)) + "|",
        ]
    )
    for row in rows:
        lines.append(
            "|" + "|".join(_fmt_md(row.get(c)) for c in TABLE_COLUMNS) + "|"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _populate_benchmark_metrics(
    row: dict[str, Any],
    bench: dict[str, Any],
) -> None:
    performance = bench.get("performance", {})
    efficiency = bench.get("efficiency", {})
    raw_metrics = performance.get("raw_metrics", {})
    row["success_rate"] = performance.get("success_rate")
    episode_successes = performance.get("episode_successes")
    if isinstance(episode_successes, list):
        row["num_successes"] = int(
            sum(int(bool(x)) for x in episode_successes)
        )
    elif isinstance(episode_successes, (int, float)):
        row["num_successes"] = int(episode_successes)
    else:
        row["num_successes"] = None
    row["eval_time_s"] = efficiency.get("evaluation_time_sec")
    row["interface_call_path"] = bench.get("model", {}).get("cost_interface")
    row["avg_return"] = _extract_metric(
        raw_metrics,
        [
            "avg_return",
            "return",
            "returns",
            "episode_return",
            "episode_returns",
            "mean_return",
        ],
    )
    row["avg_final_distance"] = _extract_metric(
        raw_metrics,
        [
            "avg_final_distance",
            "final_distance",
            "distance_to_goal",
            "goal_distance",
            "episode_final_distance",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help=f"Repeatable env flag. Valid: {', '.join(SUPPORTED_ENVS)}.",
    )
    parser.add_argument(
        "--envs",
        default="",
        help="Comma-separated envs, e.g. tworoom,pusht,ogbench_cube.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Optional global teacher checkpoint override for all envs.",
    )
    parser.add_argument(
        "--tworoom-checkpoint",
        default=DEFAULT_TEACHER_CHECKPOINTS["tworoom"],
    )
    parser.add_argument(
        "--pusht-checkpoint",
        default=DEFAULT_TEACHER_CHECKPOINTS["pusht"],
    )
    parser.add_argument(
        "--cube-checkpoint",
        default=DEFAULT_TEACHER_CHECKPOINTS["ogbench_cube"],
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--ranks", default="0.95,0.90,0.80")
    parser.add_argument("--methods", default="weight_svd")
    parser.add_argument("--num-eval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compress-if-missing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/tables/emergency_closed_loop_all.*",
    )
    args = parser.parse_args()

    envs = _parse_envs(args)
    output_base = _normalize_output_base(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    env_vars = os.environ.copy()
    env_vars.setdefault("PYTHONPATH", str(Path.cwd() / "src"))
    methods = _parse_csv_str(args.methods)
    ranks = _parse_csv_floats(args.ranks)
    all_rows: list[dict[str, Any]] = []
    for env_name in envs:
        teacher_checkpoint = _teacher_checkpoint_for_env(env_name, args)
        print(f"[setup] ensuring dataset is available for env={env_name}")
        dataset_error = None
        try:
            if not args.dry_run:
                _ = load_hdf5_dataset(env_name)
        except Exception as exc:  # pragma: no cover - runtime protection
            dataset_error = str(exc)

        teacher_tag = _teacher_tag(env_name, args.seed, args.num_eval)
        teacher_json = _benchmark_json_path(
            env_name,
            teacher_tag,
            args.seed,
            args.num_eval,
        )
        teacher_row = {
            "env": env_name,
            "tag": teacher_tag,
            "model_type": "teacher",
            "method": "teacher",
            "rank_fraction": 1.0,
            "predictor_compression_ratio": 1.0,
            "total_compression_ratio": 1.0,
            "layers_compressed": 0,
            "compression_status": "teacher_anchor",
            "num_eval": int(args.num_eval),
            "seed": int(args.seed),
            "success_rate": None,
            "num_successes": None,
            "eval_time_s": None,
            "avg_return": None,
            "avg_final_distance": None,
            "model_source": "hf_teacher",
            "checkpoint": teacher_checkpoint,
            "model_path": None,
            "interface_call_path": None,
            "status": "pending",
            "error_message": None,
        }
        if dataset_error is not None:
            teacher_row["status"] = "failed"
            teacher_row["compression_status"] = "failed"
            teacher_row["error_message"] = dataset_error
            all_rows.append(teacher_row)
            if args.fail_fast:
                raise RuntimeError(dataset_error)
        else:
            if args.skip_existing and teacher_json.exists():
                teacher_row["status"] = "skipped_existing"
            else:
                ok, _elapsed, tail = _run(
                    _benchmark_teacher_cmd(
                        env_name=env_name,
                        checkpoint=teacher_checkpoint,
                        tag=teacher_tag,
                        num_eval=args.num_eval,
                        seed=args.seed,
                        device=args.device,
                    ),
                    env=env_vars,
                    dry_run=args.dry_run,
                )
                if not ok:
                    teacher_row["status"] = "failed"
                    teacher_row["compression_status"] = "failed"
                    teacher_row["error_message"] = tail
                    all_rows.append(teacher_row)
                    if args.fail_fast:
                        raise RuntimeError(tail)
                    continue
            if args.dry_run:
                teacher_row["status"] = "dry_run"
                all_rows.append(teacher_row)
            elif teacher_json.exists():
                teacher_bench = _read_json(teacher_json)
                _populate_benchmark_metrics(teacher_row, teacher_bench)
                teacher_row["status"] = "ok"
                all_rows.append(teacher_row)
            else:
                teacher_row["status"] = "failed"
                teacher_row["compression_status"] = "failed"
                teacher_row["error_message"] = (
                    f"benchmark output not found: {teacher_json}"
                )
                all_rows.append(teacher_row)
                if args.fail_fast:
                    raise RuntimeError(str(teacher_row["error_message"]))

        for method in methods:
            for rank in ranks:
                rank_tag = rank_fraction_to_tag(rank)
                model_tag = f"{_method_prefix(env_name, method)}_{rank_tag}"
                model_path = (
                    Path("outputs/compression")
                    / env_name
                    / model_tag
                    / "compressed_model.pt"
                )
                comp_report_path = (
                    Path("outputs/compression")
                    / env_name
                    / model_tag
                    / "compression_report.json"
                )
                eval_tag = (
                    f"{model_tag}_closed_loop_seed{args.seed}_n{args.num_eval}"
                )
                bench_path = _benchmark_json_path(
                    env_name,
                    eval_tag,
                    args.seed,
                    args.num_eval,
                )
                row = {
                    "env": env_name,
                    "tag": eval_tag,
                    "model_type": "compressed",
                    "method": method,
                    "rank_fraction": float(rank),
                    "predictor_compression_ratio": None,
                    "total_compression_ratio": None,
                    "layers_compressed": None,
                    "compression_status": "pending",
                    "num_eval": int(args.num_eval),
                    "seed": int(args.seed),
                    "success_rate": None,
                    "num_successes": None,
                    "eval_time_s": None,
                    "avg_return": None,
                    "avg_final_distance": None,
                    "model_source": "local_compressed_artifact",
                    "checkpoint": teacher_checkpoint,
                    "model_path": str(model_path),
                    "interface_call_path": None,
                    "status": "pending",
                    "error_message": None,
                }

                if not model_path.exists():
                    if args.compress_if_missing:
                        ok, _elapsed, tail = _run(
                            _compression_cmd(
                                method=method,
                                env_name=env_name,
                                teacher_checkpoint=teacher_checkpoint,
                                rank_fraction=float(rank),
                                device=args.device,
                                tag=model_tag,
                            ),
                            env=env_vars,
                            dry_run=args.dry_run,
                        )
                        if not ok:
                            row["status"] = "failed"
                            row["compression_status"] = "failed"
                            row["error_message"] = tail
                            all_rows.append(row)
                            if args.fail_fast:
                                raise RuntimeError(tail)
                            continue
                    else:
                        row["status"] = "failed"
                        row["compression_status"] = "failed"
                        row["error_message"] = (
                            "compressed model missing; rerun with "
                            "--compress-if-missing"
                        )
                        all_rows.append(row)
                        if args.fail_fast:
                            raise RuntimeError(str(row["error_message"]))
                        continue

                if comp_report_path.exists():
                    comp = _read_json(comp_report_path)
                    row["predictor_compression_ratio"] = comp.get(
                        "predictor_compression_ratio"
                    )
                    row["total_compression_ratio"] = comp.get(
                        "total_compression_ratio"
                    )
                    row["layers_compressed"] = comp.get(
                        "num_layers_compressed"
                    )
                    row["compression_status"] = comp.get(
                        "compression_status", "compressed"
                    )
                    if comp.get("checkpoint"):
                        row["checkpoint"] = comp.get("checkpoint")
                elif not args.dry_run:
                    row["compression_status"] = "missing_report"

                if args.skip_existing and bench_path.exists():
                    row["status"] = "skipped_existing"
                else:
                    ok, _elapsed, tail = _run(
                        _benchmark_local_cmd(
                            env_name=env_name,
                            model_path=model_path,
                            tag=eval_tag,
                            num_eval=args.num_eval,
                            seed=args.seed,
                            device=args.device,
                        ),
                        env=env_vars,
                        dry_run=args.dry_run,
                    )
                    if not ok:
                        row["status"] = "failed"
                        row["compression_status"] = "failed"
                        row["error_message"] = tail
                        all_rows.append(row)
                        if args.fail_fast:
                            raise RuntimeError(tail)
                        continue

                if args.dry_run:
                    row["status"] = "dry_run"
                    all_rows.append(row)
                    continue
                if bench_path.exists():
                    bench = _read_json(bench_path)
                    _populate_benchmark_metrics(row, bench)
                    row["status"] = (
                        "ok"
                        if row["status"] == "pending"
                        else row["status"]
                    )
                else:
                    row["status"] = "failed"
                    row["compression_status"] = "failed"
                    row["error_message"] = (
                        f"benchmark output not found: {bench_path}"
                    )
                    if args.fail_fast:
                        raise RuntimeError(str(row["error_message"]))
                all_rows.append(row)

    combined_csv = output_base.with_suffix(".csv")
    combined_json = output_base.with_suffix(".json")
    combined_md = output_base.with_suffix(".md")
    _write_table_bundle(
        rows=all_rows,
        csv_path=combined_csv,
        json_path=combined_json,
        md_path=combined_md,
        title="Emergency Closed-Loop Frontier (All Envs)",
    )
    print(f"[done] wrote {combined_csv}")
    print(f"[done] wrote {combined_json}")
    print(f"[done] wrote {combined_md}")

    for env_name in envs:
        env_rows = [row for row in all_rows if row["env"] == env_name]
        env_base = Path("outputs/tables") / f"emergency_closed_loop_{env_name}"
        _write_table_bundle(
            rows=env_rows,
            csv_path=env_base.with_suffix(".csv"),
            json_path=env_base.with_suffix(".json"),
            md_path=env_base.with_suffix(".md"),
            title=f"Emergency Closed-Loop Frontier ({env_name})",
        )
        print(f"[done] wrote {env_base.with_suffix('.csv')}")
        print(f"[done] wrote {env_base.with_suffix('.json')}")
        print(f"[done] wrote {env_base.with_suffix('.md')}")


if __name__ == "__main__":
    main()
