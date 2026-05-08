from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from oawc.compression.operator_eval import (
    evaluate_model_on_operator_cache,
    rank_fraction_to_tag,
)
from oawc.compression.reports import count_parameters, save_json
from oawc.models import load_cost_model


DEFAULT_RANKS = [
    1.00,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
]
DEFAULT_DISTILL_RANKS = [0.90, 0.80, 0.70, 0.60, 0.50]


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_floats_csv(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _run(cmd: list[str], env: dict[str, str], dry_run: bool) -> tuple[bool, float]:
    print("[cmd]", " ".join(cmd))
    if dry_run:
        return True, 0.0
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
    for line in proc.stdout:
        print(line, end="")
    code = proc.wait()
    return code == 0, time.time() - start


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _method_tag_prefix(method: str) -> str:
    if method == "weight_svd":
        return "lewm_tworoom_svd"
    if method == "aa_svd":
        return "lewm_tworoom_aa_svd"
    raise ValueError(f"Unknown method: {method}")


def _compress_cmd(
    *,
    method: str,
    env: str,
    checkpoint: str,
    rank_fraction: float,
    device: str,
    tag: str,
    num_calib_states: int,
    num_calib_candidates: int,
    calib_batch_states: int,
    calib_batch_candidates: int,
) -> list[str]:
    if method == "weight_svd":
        return [
            "uv",
            "run",
            "python",
            "scripts/compress_predictor_weight_svd.py",
            "--env",
            env,
            "--model-family",
            "lewm_hf",
            "--checkpoint",
            checkpoint,
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
        env,
        "--model-family",
        "lewm_hf",
        "--checkpoint",
        checkpoint,
        "--rank-fraction",
        str(rank_fraction),
        "--target-substring",
        "predictor",
        "--min-dim",
        "64",
        "--num-calib-states",
        str(num_calib_states),
        "--num-calib-candidates",
        str(num_calib_candidates),
        "--horizon",
        "5",
        "--max-rows-per-layer",
        "16384",
        "--ridge",
        "1e-4",
        "--calib-batch-states",
        str(calib_batch_states),
        "--calib-batch-candidates",
        str(calib_batch_candidates),
        "--device",
        device,
        "--tag",
        tag,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument("--teacher-checkpoint", default="quentinll/lewm-tworooms")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--rank-fractions",
        default=",".join(str(x) for x in DEFAULT_RANKS),
    )
    parser.add_argument("--methods", default="weight_svd,aa_svd")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--copy-to-drive", type=_str2bool, default=False)
    parser.add_argument("--drive-root", default=None)
    parser.add_argument("--num-calib-states", type=int, default=128)
    parser.add_argument("--num-calib-candidates", type=int, default=128)
    parser.add_argument("--calib-batch-states", type=int, default=8)
    parser.add_argument("--calib-batch-candidates", type=int, default=128)
    parser.add_argument(
        "--eval-cache",
        default=(
            "outputs/operator_cache/tworoom/"
            "lewm_tworoom_eval_s128_c128_seed1/operator_cache.pt"
        ),
    )
    parser.add_argument(
        "--train-cache",
        default=(
            "outputs/operator_cache/tworoom/"
            "lewm_tworoom_train_s512_c128_seed0/operator_cache.pt"
        ),
    )
    parser.add_argument("--eval-random-cache", type=_str2bool, default=True)
    parser.add_argument("--run-distill-subset", type=_str2bool, default=False)
    parser.add_argument(
        "--distill-ranks",
        default=",".join(str(x) for x in DEFAULT_DISTILL_RANKS),
    )
    parser.add_argument("--distill-steps", type=int, default=100)
    parser.add_argument("--distill-batch-size", type=int, default=8)
    parser.add_argument("--distill-lr", type=float, default=1e-5)
    parser.add_argument("--identity-min-spearman", type=float, default=0.99)
    parser.add_argument("--identity-max-mse", type=float, default=1e-3)
    parser.add_argument("--identity-min-top1", type=float, default=0.99)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ranks = sorted(_parse_floats_csv(args.rank_fractions), reverse=True)
    distill_ranks = set(round(x, 2) for x in _parse_floats_csv(args.distill_ranks))
    env_vars = os.environ.copy()
    env_vars.setdefault("PYTHONPATH", str(Path.cwd() / "src"))

    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "env": args.env,
        "teacher_checkpoint": args.teacher_checkpoint,
        "device": args.device,
        "methods": methods,
        "rank_fractions": ranks,
        "eval_cache": args.eval_cache,
        "train_cache": args.train_cache,
        "teacher_anchor_source": "local_artifact",
        "identity_check_passed": False,
        "identity_details": None,
        "runs": [],
    }

    # Teacher anchor for canonical random-cache evaluation.
    if bool(args.eval_random_cache):
        teacher_tag = "lewm_tworoom_teacher_r100"
        teacher_dir = Path("outputs/compression") / args.env / teacher_tag
        teacher_dir.mkdir(parents=True, exist_ok=True)
        teacher_model_path = teacher_dir / "compressed_model.pt"
        teacher_report = teacher_dir / "compression_report.json"
        if args.dry_run:
            print(
                "[dry-run] would materialize teacher anchor model at "
                f"{teacher_model_path}"
            )
        elif not (args.skip_existing and teacher_model_path.exists()):
            loaded = load_cost_model(
                family="lewm_hf",
                checkpoint=args.teacher_checkpoint,
                env_name=args.env,
                device="cpu",
            )
            teacher_model = loaded.model.to("cpu").eval()
            teacher_model.requires_grad_(False)
            torch.save(teacher_model, teacher_model_path)
            total_params = int(count_parameters(teacher_model))
            save_json(
                teacher_report,
                {
                    "method": "teacher_anchor",
                    "tag": teacher_tag,
                    "rank_fraction": 1.0,
                    "predictor_compression_ratio": 1.0,
                    "total_compression_ratio": 1.0,
                    "total_params_before": total_params,
                    "total_params_after": total_params,
                    "predictor_params_before": None,
                    "predictor_params_after": None,
                    "compressed_model_path": str(teacher_model_path),
                },
            )
        if not args.dry_run:
            hf_loaded = load_cost_model(
                family="lewm_hf",
                checkpoint=args.teacher_checkpoint,
                env_name=args.env,
                device="cpu",
            )
            hf_result = evaluate_model_on_operator_cache(
                cache_path=args.eval_cache,
                model=hf_loaded.model.to("cpu").eval(),
                model_path=args.teacher_checkpoint,
                device="cpu",
                use_chunked_student=False,
            )
            local_result = evaluate_model_on_operator_cache(
                cache_path=args.eval_cache,
                model_path=str(teacher_model_path),
                device="cpu",
                use_chunked_student=False,
            )
            ident = {
                "hf_direct": {
                    "raw_cost_mse": float(hf_result["metrics"]["raw_cost_mse"]),
                    "spearman_mean": float(
                        hf_result["metrics"]["spearman_per_state"]["mean"]
                    ),
                    "top1_match_rate": float(
                        hf_result["metrics"]["teacher_best_index_match_rate"]
                    ),
                },
                "local_artifact": {
                    "raw_cost_mse": float(local_result["metrics"]["raw_cost_mse"]),
                    "spearman_mean": float(
                        local_result["metrics"]["spearman_per_state"]["mean"]
                    ),
                    "top1_match_rate": float(
                        local_result["metrics"]["teacher_best_index_match_rate"]
                    ),
                },
            }
            passed = bool(
                ident["local_artifact"]["spearman_mean"]
                >= float(args.identity_min_spearman)
                and ident["local_artifact"]["raw_cost_mse"]
                <= float(args.identity_max_mse)
                and ident["local_artifact"]["top1_match_rate"]
                >= float(args.identity_min_top1)
            )
            manifest["identity_details"] = ident
            manifest["identity_check_passed"] = passed
            if not passed:
                out_path = Path(
                    "outputs/tables/rank_frontier_run_manifest_tworoom.json"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError(
                    "Identity invariant failed for local teacher r100 artifact: "
                    f"{ident['local_artifact']}. "
                    "Aborting frontier run."
                )
        teacher_eval_tag = "lewm_tworoom_teacher_r100_eval_s128_seed1"
        teacher_metrics = (
            Path("outputs/operator_metrics") / args.env / teacher_eval_tag / "metrics.json"
        )
        if not (args.skip_existing and teacher_metrics.exists()):
            ok, elapsed = _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/evaluate_operator_metrics.py",
                    "--cache",
                    args.eval_cache,
                    "--model-path",
                    str(teacher_model_path),
                    "--device",
                    args.device,
                    "--tag",
                    teacher_eval_tag,
                ],
                env=env_vars,
                dry_run=args.dry_run,
            )
            manifest["runs"].append(
                {
                    "kind": "teacher_eval_random",
                    "tag": teacher_eval_tag,
                    "ok": ok,
                    "elapsed_sec": elapsed,
                }
            )

    for method in methods:
        prefix = _method_tag_prefix(method)
        for rank in ranks:
            rtag = rank_fraction_to_tag(rank)
            base_tag = f"{prefix}_{rtag}"
            entry: dict[str, Any] = {
                "kind": "compression",
                "method": method,
                "rank_fraction": rank,
                "tag": base_tag,
            }
            if rank >= 0.999:
                entry["status"] = "teacher_anchor"
                manifest["runs"].append(entry)
                continue

            model_path = (
                Path("outputs/compression") / args.env / base_tag / "compressed_model.pt"
            )
            if args.skip_existing and model_path.exists():
                entry["status"] = "skipped_existing_compression"
                manifest["runs"].append(entry)
            else:
                ok, elapsed = _run(
                    _compress_cmd(
                        method=method,
                        env=args.env,
                        checkpoint=args.teacher_checkpoint,
                        rank_fraction=rank,
                        device=args.device,
                        tag=base_tag,
                        num_calib_states=int(args.num_calib_states),
                        num_calib_candidates=int(args.num_calib_candidates),
                        calib_batch_states=int(args.calib_batch_states),
                        calib_batch_candidates=int(args.calib_batch_candidates),
                    ),
                    env=env_vars,
                    dry_run=args.dry_run,
                )
                entry.update({"ok": ok, "elapsed_sec": elapsed})
                manifest["runs"].append(entry)
                if not ok:
                    continue

            if bool(args.eval_random_cache):
                eval_tag = f"{base_tag}_eval_s128_seed1"
                metrics_path = (
                    Path("outputs/operator_metrics") / args.env / eval_tag / "metrics.json"
                )
                if not (args.skip_existing and metrics_path.exists()):
                    ok, elapsed = _run(
                        [
                            "uv",
                            "run",
                            "python",
                            "scripts/evaluate_operator_metrics.py",
                            "--cache",
                            args.eval_cache,
                            "--model-path",
                            str(model_path),
                            "--device",
                            args.device,
                            "--tag",
                            eval_tag,
                        ],
                        env=env_vars,
                        dry_run=args.dry_run,
                    )
                    manifest["runs"].append(
                        {
                            "kind": "random_operator_eval",
                            "tag": eval_tag,
                            "model_path": str(model_path),
                            "cache_path": args.eval_cache,
                            "candidate_mode": "random",
                            "ok": ok,
                            "elapsed_sec": elapsed,
                        }
                    )

            if bool(args.run_distill_subset) and round(rank, 2) in distill_ranks:
                for method_name, script_name in [
                    ("cost_kl", "distill_operator_cost_kl.py"),
                    ("hybrid", "distill_operator_hybrid.py"),
                ]:
                    distill_tag = f"{base_tag}_{method_name}_split"
                    out_model = (
                        Path("outputs/compression")
                        / args.env
                        / distill_tag
                        / "distilled_model.pt"
                    )
                    if args.skip_existing and out_model.exists():
                        continue
                    cmd = [
                        "uv",
                        "run",
                        "python",
                        f"scripts/{script_name}",
                        "--env",
                        args.env,
                        "--train-cache",
                        args.train_cache,
                        "--eval-cache",
                        args.eval_cache,
                        "--student-path",
                        str(model_path),
                        "--max-steps",
                        str(args.distill_steps),
                        "--batch-size",
                        str(args.distill_batch_size),
                        "--lr",
                        str(args.distill_lr),
                        "--device",
                        args.device,
                        "--val-states",
                        "64",
                        "--val-candidates",
                        "128",
                        "--save-best-by",
                        "val_top5",
                        "--eval-every",
                        "25",
                        "--tag",
                        distill_tag,
                    ]
                    ok, elapsed = _run(cmd, env=env_vars, dry_run=args.dry_run)
                    manifest["runs"].append(
                        {
                            "kind": "distill",
                            "distill_method": method_name,
                            "tag": distill_tag,
                            "base_model_tag": base_tag,
                            "ok": ok,
                            "elapsed_sec": elapsed,
                        }
                    )

    manifest_path = Path("outputs/tables/rank_frontier_run_manifest_tworoom.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote manifest: {manifest_path}")

    if bool(args.copy_to_drive):
        if not args.drive_root:
            raise ValueError("--drive-root is required when --copy-to-drive true")
        drive_root = Path(args.drive_root)
        targets = [
            Path("outputs/compression") / args.env,
            Path("outputs/operator_metrics") / args.env,
            Path("outputs/tables"),
        ]
        for src in targets:
            if src.exists():
                _copy_path(src, drive_root / src)
                print(f"[copy] {src} -> {drive_root / src}")


if __name__ == "__main__":
    main()
