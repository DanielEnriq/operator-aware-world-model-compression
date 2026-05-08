from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oawc.benchmark import load_hdf5_dataset
from oawc.compression.operator_eval import rank_fraction_to_tag


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


def _run(
    cmd: list[str],
    env: dict[str, str],
) -> tuple[bool, float, str]:
    print("[cmd]", " ".join(cmd))
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


def _teacher_tag(seed: int, num_eval: int) -> str:
    return f"lewm_tworoom_teacher_closed_loop_seed{seed}_n{num_eval}"


def _method_prefix(method: str) -> str:
    if method == "weight_svd":
        return "lewm_tworoom_svd"
    if method == "aa_svd":
        return "lewm_tworoom_aa_svd"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument(
        "--teacher-checkpoint",
        default="quentinll/lewm-tworooms",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--ranks", default="0.95,0.90,0.80,0.70,0.60,0.50")
    parser.add_argument("--methods", default="weight_svd,aa_svd")
    parser.add_argument("--num-eval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--compress-if-missing", action="store_true")
    parser.add_argument(
        "--output",
        default="outputs/tables/emergency_closed_loop_tworoom.*",
    )
    args = parser.parse_args()

    output_base = _normalize_output_base(args.output)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    env_vars = os.environ.copy()
    env_vars.setdefault("PYTHONPATH", str(Path.cwd() / "src"))

    print("[setup] ensuring dataset is available")
    _ = load_hdf5_dataset(args.env)

    rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    methods = _parse_csv_str(args.methods)
    ranks = _parse_csv_floats(args.ranks)

    teacher_tag = _teacher_tag(args.seed, args.num_eval)
    teacher_json = _benchmark_json_path(
        args.env, teacher_tag, args.seed, args.num_eval
    )
    teacher_row: dict[str, Any] = {
        "timestamp_utc": now,
        "tag": teacher_tag,
        "method": "teacher",
        "rank_fraction": 1.0,
        "model_source": "hf_teacher",
        "predictor_param_ratio": 1.0,
        "total_param_ratio": 1.0,
        "compression_status": "teacher_anchor",
        "success_rate": None,
        "num_eval": int(args.num_eval),
        "seed": int(args.seed),
        "eval_time_s": None,
        "model_path": args.teacher_checkpoint,
        "benchmark_json": str(teacher_json),
        "compression_report": None,
        "notes": "",
    }
    if not (args.skip_existing and teacher_json.exists()):
        ok, elapsed, tail = _run(
            _benchmark_teacher_cmd(
                env_name=args.env,
                checkpoint=args.teacher_checkpoint,
                tag=teacher_tag,
                num_eval=args.num_eval,
                seed=args.seed,
                device=args.device,
            ),
            env=env_vars,
        )
        if not ok:
            teacher_row["compression_status"] = "failed"
            teacher_row["notes"] = (
                f"teacher benchmark failed (elapsed={elapsed:.2f}s): {tail}"
            )
    if teacher_json.exists():
        bench = _read_json(teacher_json)
        teacher_row["success_rate"] = bench.get("performance", {}).get(
            "success_rate"
        )
        teacher_row["eval_time_s"] = bench.get("efficiency", {}).get(
            "evaluation_time_sec"
        )
    elif teacher_row["compression_status"] != "failed":
        teacher_row["compression_status"] = "failed"
        teacher_row["notes"] = "teacher benchmark output not found"
    rows.append(teacher_row)

    for method in methods:
        for rank in ranks:
            rank_tag = rank_fraction_to_tag(rank)
            model_tag = f"{_method_prefix(method)}_{rank_tag}"
            model_path = (
                Path("outputs/compression")
                / args.env
                / model_tag
                / "compressed_model.pt"
            )
            comp_report_path = (
                Path("outputs/compression")
                / args.env
                / model_tag
                / "compression_report.json"
            )
            eval_tag = (
                f"{model_tag}_closed_loop_seed{args.seed}_n{args.num_eval}"
            )
            bench_path = _benchmark_json_path(
                args.env, eval_tag, args.seed, args.num_eval
            )
            row: dict[str, Any] = {
                "timestamp_utc": now,
                "tag": eval_tag,
                "method": method,
                "rank_fraction": float(rank),
                "model_source": "local_compressed_artifact",
                "predictor_param_ratio": None,
                "total_param_ratio": None,
                "compression_status": "pending",
                "success_rate": None,
                "num_eval": int(args.num_eval),
                "seed": int(args.seed),
                "eval_time_s": None,
                "model_path": str(model_path),
                "benchmark_json": str(bench_path),
                "compression_report": (
                    str(comp_report_path)
                    if comp_report_path.exists()
                    else None
                ),
                "notes": "",
            }

            if not model_path.exists():
                if args.compress_if_missing:
                    ok, elapsed, tail = _run(
                        _compression_cmd(
                            method=method,
                            env_name=args.env,
                            teacher_checkpoint=args.teacher_checkpoint,
                            rank_fraction=float(rank),
                            device=args.device,
                            tag=model_tag,
                        ),
                        env=env_vars,
                    )
                    if not ok:
                        row["compression_status"] = "failed"
                        row["notes"] = (
                            "compression failed "
                            f"(elapsed={elapsed:.2f}s): {tail}"
                        )
                        rows.append(row)
                        continue
                else:
                    row["compression_status"] = "failed"
                    row["notes"] = (
                        "compressed model missing; rerun with "
                        "--compress-if-missing"
                    )
                    rows.append(row)
                    continue

            if comp_report_path.exists():
                comp = _read_json(comp_report_path)
                row["predictor_param_ratio"] = comp.get(
                    "predictor_compression_ratio"
                )
                row["total_param_ratio"] = comp.get("total_compression_ratio")
                row["compression_status"] = comp.get(
                    "compression_status", "compressed"
                )
                row["compression_report"] = str(comp_report_path)
            else:
                row["compression_status"] = "missing_report"
                row["notes"] = "compression_report.json not found"

            if not (args.skip_existing and bench_path.exists()):
                ok, elapsed, tail = _run(
                    _benchmark_local_cmd(
                        env_name=args.env,
                        model_path=model_path,
                        tag=eval_tag,
                        num_eval=args.num_eval,
                        seed=args.seed,
                        device=args.device,
                    ),
                    env=env_vars,
                )
                if not ok:
                    row["compression_status"] = "failed"
                    row["notes"] = (
                        f"benchmark failed (elapsed={elapsed:.2f}s): {tail}"
                    )
            if bench_path.exists():
                bench = _read_json(bench_path)
                row["success_rate"] = bench.get("performance", {}).get(
                    "success_rate"
                )
                row["eval_time_s"] = bench.get("efficiency", {}).get(
                    "evaluation_time_sec"
                )
            elif row["compression_status"] != "failed":
                row["compression_status"] = "failed"
                row["notes"] = "benchmark output not found"

            rows.append(row)

    columns = [
        "tag",
        "method",
        "rank_fraction",
        "model_source",
        "predictor_param_ratio",
        "total_param_ratio",
        "compression_status",
        "success_rate",
        "num_eval",
        "seed",
        "eval_time_s",
        "model_path",
        "benchmark_json",
        "compression_report",
        "notes",
    ]

    csv_path = output_base.with_suffix(".csv")
    json_path = output_base.with_suffix(".json")
    md_path = output_base.with_suffix(".md")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})

    payload = {
        "timestamp_utc": now,
        "env": args.env,
        "teacher_checkpoint": args.teacher_checkpoint,
        "num_eval": int(args.num_eval),
        "seed": int(args.seed),
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    warning = (
        "Primary metric is closed-loop MPC success. "
        "Operator-cache metrics are not used here because the local r100 "
        "identity invariant previously failed. Teacher anchor is evaluated "
        "directly from HF checkpoint, not from a local no-op artifact."
    )
    lines = [
        "# Emergency Closed-Loop Frontier (TwoRoom)",
        "",
        f"> **Warning:** {warning}",
        "",
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        lines.append(
            "|" + "|".join(_fmt_md(row.get(c)) for c in columns) + "|"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {md_path}")


if __name__ == "__main__":
    main()
