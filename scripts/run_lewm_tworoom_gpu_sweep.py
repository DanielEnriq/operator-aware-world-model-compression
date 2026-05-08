from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", required=True)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--env", default="tworoom")
    parser.add_argument(
        "--teacher-checkpoint",
        default="quentinll/lewm-tworooms",
    )
    parser.add_argument("--train-states", type=int, default=512)
    parser.add_argument("--eval-states", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed-train", type=int, default=0)
    parser.add_argument("--seed-eval", type=int, default=1)
    parser.add_argument("--distill-steps", type=int, default=100)
    parser.add_argument("--distill-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--cost-batch-states", type=int, default=8)
    parser.add_argument("--cost-batch-candidates", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--copy-to-drive", type=_str2bool, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _run_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    dry_run: bool,
) -> tuple[bool, float]:
    cmd_str = " ".join(cmd)
    print(f"[cmd] {cmd_str}")
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
    elapsed = time.time() - start
    return code == 0, elapsed


def _path_exists(path: Path) -> bool:
    return path.exists()


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _stage_enabled(
    stage_key: str,
    stage_letter: str,
    only: str | None,
) -> bool:
    if only is None:
        return True
    only_norm = only.strip().lower()
    return only_norm in {stage_key.lower(), stage_letter.lower()}


def _stage_banner(letter: str, key: str, when: str) -> None:
    print(f"[stage {letter}] {key} {when}")


def _env_check(env: dict[str, str], project_root: Path) -> dict[str, Any]:
    import importlib

    import torch

    swm = importlib.import_module("stable_worldmodel")
    swm_file = str(Path(swm.__file__).resolve())
    if "external/stable-worldmodel" in swm_file:
        raise RuntimeError(
            "Installed-lane sweep requires installed stable_worldmodel, got "
            f"{swm_file}"
        )

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    info = {
        "cwd": str(project_root),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "PYTHONPATH": env.get("PYTHONPATH"),
        "STABLEWM_HOME": env.get("STABLEWM_HOME"),
        "stable_worldmodel_file": swm_file,
    }
    print(json.dumps(info, indent=2))
    return info


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}/src"
    env["STABLEWM_HOME"] = str(project_root / ".swm_cache")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    Path(env["STABLEWM_HOME"]).mkdir(parents=True, exist_ok=True)

    train_tag = (
        f"lewm_tworoom_train_s{args.train_states}_c{args.num_candidates}_"
        f"seed{args.seed_train}"
    )
    eval_tag = (
        f"lewm_tworoom_eval_s{args.eval_states}_c{args.num_candidates}_"
        f"seed{args.seed_eval}"
    )
    train_cache = (
        Path("outputs/operator_cache")
        / args.env
        / train_tag
        / "operator_cache.pt"
    )
    eval_cache = (
        Path("outputs/operator_cache")
        / args.env
        / eval_tag
        / "operator_cache.pt"
    )

    completed_stages: list[str] = []
    failed_stages: list[dict[str, Any]] = []
    stage_timings: dict[str, float] = {}

    def run_stage(
        *,
        letter: str,
        key: str,
        fn,
    ) -> bool:
        if not _stage_enabled(key, letter, args.only):
            print(f"[stage {letter}] {key} skipped (--only={args.only})")
            return True
        _stage_banner(letter, key, "start")
        start = time.time()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            stage_timings[key] = elapsed
            failed_stages.append(
                {
                    "stage": key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_sec": elapsed,
                }
            )
            print(f"[stage {letter}] {key} failed: {exc}")
            if not args.continue_on_error:
                return False
            return True
        elapsed = time.time() - start
        stage_timings[key] = elapsed
        completed_stages.append(key)
        _stage_banner(letter, key, f"finish ({elapsed:.1f}s)")
        return True

    env_info: dict[str, Any] = {}

    def stage_a() -> None:
        nonlocal env_info
        env_info = _env_check(env, project_root)

    def stage_b() -> None:
        train_cache_dir = train_cache.parent
        eval_cache_dir = eval_cache.parent
        if (
            args.skip_existing
            and _path_exists(train_cache)
            and _path_exists(eval_cache)
        ):
            print("[skip] train/eval caches already exist")
            return
        train_cache_dir.mkdir(parents=True, exist_ok=True)
        eval_cache_dir.mkdir(parents=True, exist_ok=True)
        ok, _ = _run_command(
            [
                "uv",
                "run",
                "python",
                "scripts/build_operator_cache.py",
                "--env",
                args.env,
                "--model-family",
                "lewm_hf",
                "--checkpoint",
                args.teacher_checkpoint,
                "--num-states",
                str(args.train_states),
                "--num-candidates",
                str(args.num_candidates),
                "--horizon",
                str(args.horizon),
                "--topk",
                "1",
                "5",
                "10",
                "20",
                "--seed",
                str(args.seed_train),
                "--device",
                args.device,
                "--tag",
                train_tag,
                "--split",
                "train",
                "--cost-batch-states",
                str(args.cost_batch_states),
                "--cost-batch-candidates",
                str(args.cost_batch_candidates),
            ],
            env=env,
            dry_run=args.dry_run,
        )
        if not ok:
            raise RuntimeError("Train cache build failed.")
        ok, _ = _run_command(
            [
                "uv",
                "run",
                "python",
                "scripts/build_operator_cache.py",
                "--env",
                args.env,
                "--model-family",
                "lewm_hf",
                "--checkpoint",
                args.teacher_checkpoint,
                "--num-states",
                str(args.eval_states),
                "--num-candidates",
                str(args.num_candidates),
                "--horizon",
                str(args.horizon),
                "--topk",
                "1",
                "5",
                "10",
                "20",
                "--seed",
                str(args.seed_eval),
                "--device",
                args.device,
                "--tag",
                eval_tag,
                "--split",
                "eval",
                "--cost-batch-states",
                str(args.cost_batch_states),
                "--cost-batch-candidates",
                str(args.cost_batch_candidates),
            ],
            env=env,
            dry_run=args.dry_run,
        )
        if not ok:
            raise RuntimeError("Eval cache build failed.")

    def stage_c() -> None:
        for cache in [train_cache, eval_cache]:
            ok, _ = _run_command(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/check_operator_cache.py",
                    str(cache),
                    "--summary-json",
                ],
                env=env,
                dry_run=args.dry_run,
            )
            if not ok:
                raise RuntimeError(f"Cache check failed: {cache}")

    def stage_d() -> None:
        jobs = [
            (
                "lewm_tworoom_svd_r050",
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/compress_predictor_weight_svd.py",
                    "--env",
                    args.env,
                    "--model-family",
                    "lewm_hf",
                    "--checkpoint",
                    args.teacher_checkpoint,
                    "--rank-fraction",
                    "0.5",
                    "--target-substring",
                    "predictor",
                    "--min-dim",
                    "64",
                    "--device",
                    args.device,
                    "--tag",
                    "lewm_tworoom_svd_r050",
                ],
                Path(
                    "outputs/compression/tworoom/"
                    "lewm_tworoom_svd_r050/compressed_model.pt"
                ),
            ),
            (
                "lewm_tworoom_svd_r025",
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/compress_predictor_weight_svd.py",
                    "--env",
                    args.env,
                    "--model-family",
                    "lewm_hf",
                    "--checkpoint",
                    args.teacher_checkpoint,
                    "--rank-fraction",
                    "0.25",
                    "--target-substring",
                    "predictor",
                    "--min-dim",
                    "64",
                    "--device",
                    args.device,
                    "--tag",
                    "lewm_tworoom_svd_r025",
                ],
                Path(
                    "outputs/compression/tworoom/"
                    "lewm_tworoom_svd_r025/compressed_model.pt"
                ),
            ),
            (
                "lewm_tworoom_aa_svd_r050",
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/compress_predictor_activation_svd.py",
                    "--env",
                    args.env,
                    "--model-family",
                    "lewm_hf",
                    "--checkpoint",
                    args.teacher_checkpoint,
                    "--rank-fraction",
                    "0.5",
                    "--target-substring",
                    "predictor",
                    "--min-dim",
                    "64",
                    "--num-calib-states",
                    str(args.train_states),
                    "--num-calib-candidates",
                    str(args.num_candidates),
                    "--horizon",
                    str(args.horizon),
                    "--max-rows-per-layer",
                    "16384",
                    "--ridge",
                    "1e-4",
                    "--device",
                    args.device,
                    "--tag",
                    "lewm_tworoom_aa_svd_r050",
                ],
                Path(
                    "outputs/compression/tworoom/"
                    "lewm_tworoom_aa_svd_r050/compressed_model.pt"
                ),
            ),
            (
                "lewm_tworoom_aa_svd_r025",
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/compress_predictor_activation_svd.py",
                    "--env",
                    args.env,
                    "--model-family",
                    "lewm_hf",
                    "--checkpoint",
                    args.teacher_checkpoint,
                    "--rank-fraction",
                    "0.25",
                    "--target-substring",
                    "predictor",
                    "--min-dim",
                    "64",
                    "--num-calib-states",
                    str(args.train_states),
                    "--num-calib-candidates",
                    str(args.num_candidates),
                    "--horizon",
                    str(args.horizon),
                    "--max-rows-per-layer",
                    "16384",
                    "--ridge",
                    "1e-4",
                    "--device",
                    args.device,
                    "--tag",
                    "lewm_tworoom_aa_svd_r025",
                ],
                Path(
                    "outputs/compression/tworoom/"
                    "lewm_tworoom_aa_svd_r025/compressed_model.pt"
                ),
            ),
        ]
        for name, cmd, out_path in jobs:
            if args.skip_existing and out_path.exists():
                print(f"[skip] {name} exists: {out_path}")
                continue
            ok, _ = _run_command(cmd, env=env, dry_run=args.dry_run)
            if not ok:
                raise RuntimeError(f"Compression job failed: {name}")

    def stage_e() -> None:
        eval_jobs = [
            (
                "lewm_tworoom_svd_r050",
                "lewm_tworoom_svd_r050_eval_s128_seed1",
            ),
            (
                "lewm_tworoom_svd_r025",
                "lewm_tworoom_svd_r025_eval_s128_seed1",
            ),
            (
                "lewm_tworoom_aa_svd_r050",
                "lewm_tworoom_aa_svd_r050_eval_s128_seed1",
            ),
            (
                "lewm_tworoom_aa_svd_r025",
                "lewm_tworoom_aa_svd_r025_eval_s128_seed1",
            ),
        ]
        for base_tag, eval_tag_name in eval_jobs:
            model_path = (
                Path("outputs/compression")
                / args.env
                / base_tag
                / "compressed_model.pt"
            )
            out_metrics = (
                Path("outputs/operator_metrics")
                / args.env
                / eval_tag_name
                / "metrics.json"
            )
            if args.skip_existing and out_metrics.exists():
                print(f"[skip] held-out baseline metrics exist: {out_metrics}")
                continue
            ok, _ = _run_command(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/evaluate_operator_metrics.py",
                    "--cache",
                    str(eval_cache),
                    "--model-path",
                    str(model_path),
                    "--device",
                    args.device,
                    "--tag",
                    eval_tag_name,
                ],
                env=env,
                dry_run=args.dry_run,
            )
            if not ok:
                raise RuntimeError(
                    "Held-out baseline evaluation failed: "
                    f"{base_tag}"
                )

    def stage_f() -> None:
        distill_jobs = [
            (
                "lewm_tworoom_svd_r050_cost_kl_split",
                "scripts/distill_operator_cost_kl.py",
                (
                    "outputs/compression/tworoom/"
                    "lewm_tworoom_svd_r050/compressed_model.pt"
                ),
                [],
            ),
            (
                "lewm_tworoom_svd_r050_elite_k10_split",
                "scripts/distill_operator_elite.py",
                (
                    "outputs/compression/tworoom/"
                    "lewm_tworoom_svd_r050/compressed_model.pt"
                ),
                ["--elite-k", "10", "--elite-loss", "balanced_bce"],
            ),
            (
                "lewm_tworoom_svd_r050_hybrid_split",
                "scripts/distill_operator_hybrid.py",
                (
                    "outputs/compression/tworoom/"
                    "lewm_tworoom_svd_r050/compressed_model.pt"
                ),
                [
                    "--lambda-kl",
                    "1.0",
                    "--lambda-elite",
                    "1.0",
                    "--lambda-pred",
                    "0.0",
                    "--tau-kl",
                    "1.0",
                    "--tau-elite",
                    "1.0",
                    "--elite-k",
                    "10",
                    "--elite-loss",
                    "balanced_bce",
                ],
            ),
            (
                "lewm_tworoom_aa_svd_r025_hybrid_split",
                "scripts/distill_operator_hybrid.py",
                (
                    "outputs/compression/tworoom/"
                    "lewm_tworoom_aa_svd_r025/compressed_model.pt"
                ),
                [
                    "--lambda-kl",
                    "1.0",
                    "--lambda-elite",
                    "1.0",
                    "--lambda-pred",
                    "0.0",
                    "--tau-kl",
                    "1.0",
                    "--tau-elite",
                    "1.0",
                    "--elite-k",
                    "10",
                    "--elite-loss",
                    "balanced_bce",
                ],
            ),
            (
                "lewm_tworoom_svd_r025_hybrid_split",
                "scripts/distill_operator_hybrid.py",
                (
                    "outputs/compression/tworoom/"
                    "lewm_tworoom_svd_r025/compressed_model.pt"
                ),
                [
                    "--lambda-kl",
                    "1.0",
                    "--lambda-elite",
                    "1.0",
                    "--lambda-pred",
                    "0.0",
                    "--tau-kl",
                    "1.0",
                    "--tau-elite",
                    "1.0",
                    "--elite-k",
                    "10",
                    "--elite-loss",
                    "balanced_bce",
                ],
            ),
        ]
        for tag, script, student_path, extra in distill_jobs:
            out_model = (
                (
                    Path("outputs/compression")
                    / args.env
                    / tag
                    / "distilled_model.pt"
                )
            )
            if args.skip_existing and out_model.exists():
                print(f"[skip] distillation output exists: {out_model}")
                continue
            cmd = [
                "uv",
                "run",
                "python",
                script,
                "--env",
                args.env,
                "--train-cache",
                str(train_cache),
                "--eval-cache",
                str(eval_cache),
                "--student-path",
                student_path,
                "--max-steps",
                str(args.distill_steps),
                "--batch-size",
                str(args.distill_batch_size),
                "--lr",
                str(args.lr),
                "--normalize-costs",
                "zscore",
                "--trainable-substring",
                "predictor",
                "--device",
                args.device,
                "--tag",
                tag,
            ] + extra
            ok, _ = _run_command(cmd, env=env, dry_run=args.dry_run)
            if not ok:
                raise RuntimeError(f"Distillation job failed: {tag}")

    def stage_g() -> None:
        eval_jobs = [
            (
                "lewm_tworoom_svd_r050_cost_kl_split",
                "lewm_tworoom_svd_r050_cost_kl_split_eval_s128_seed1",
            ),
            (
                "lewm_tworoom_svd_r050_elite_k10_split",
                "lewm_tworoom_svd_r050_elite_k10_split_eval_s128_seed1",
            ),
            (
                "lewm_tworoom_svd_r050_hybrid_split",
                "lewm_tworoom_svd_r050_hybrid_split_eval_s128_seed1",
            ),
            (
                "lewm_tworoom_aa_svd_r025_hybrid_split",
                "lewm_tworoom_aa_svd_r025_hybrid_split_eval_s128_seed1",
            ),
            (
                "lewm_tworoom_svd_r025_hybrid_split",
                "lewm_tworoom_svd_r025_hybrid_split_eval_s128_seed1",
            ),
        ]
        for tag, eval_tag_name in eval_jobs:
            model_path = (
                Path("outputs/compression")
                / args.env
                / tag
                / "distilled_model.pt"
            )
            if not model_path.exists() and not args.dry_run:
                print(f"[skip] no distilled model for {tag}: {model_path}")
                continue
            out_metrics = (
                Path("outputs/operator_metrics")
                / args.env
                / eval_tag_name
                / "metrics.json"
            )
            if args.skip_existing and out_metrics.exists():
                print(
                    "[skip] held-out distilled metrics exist: "
                    f"{out_metrics}"
                )
                continue
            ok, _ = _run_command(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/evaluate_operator_metrics.py",
                    "--cache",
                    str(eval_cache),
                    "--model-path",
                    str(model_path),
                    "--device",
                    args.device,
                    "--tag",
                    eval_tag_name,
                ],
                env=env,
                dry_run=args.dry_run,
            )
            if not ok:
                raise RuntimeError(
                    "Held-out distilled evaluation failed: "
                    f"{tag}"
                )

    def stage_h() -> None:
        ok, _ = _run_command(
            [
                "uv",
                "run",
                "python",
                "scripts/summarize_split_operator_results.py",
                "--env",
                args.env,
                "--eval-cache-tag",
                eval_tag,
            ],
            env=env,
            dry_run=args.dry_run,
        )
        if not ok:
            raise RuntimeError("Split summary script failed.")

    def stage_i() -> None:
        ok, _ = _run_command(
            [
                "uv",
                "run",
                "python",
                "scripts/plot_split_operator_results.py",
                "--summary-json",
                "outputs/tables/operator_split_summary_tworoom.json",
                "--output-dir",
                "outputs/figures/compression_split",
            ],
            env=env,
            dry_run=args.dry_run,
        )
        if not ok:
            raise RuntimeError("Split plotting script failed.")

    def stage_j() -> None:
        candidates = Path("outputs/tables/final_benchmark_candidates.json")
        if not args.dry_run and not candidates.exists():
            raise RuntimeError(
                "final_benchmark_candidates.json missing; run stage H first."
            )
        print(f"[info] final candidates: {candidates}")

    copied_paths: list[str] = []

    def stage_k() -> None:
        if not args.copy_to_drive:
            print("[skip] drive copy disabled (--copy-to-drive false)")
            return
        drive_root = Path(args.drive_root)
        drive_root.mkdir(parents=True, exist_ok=True)
        targets = [
            Path("outputs/operator_cache") / args.env / train_tag,
            Path("outputs/operator_cache") / args.env / eval_tag,
            Path("outputs/compression") / args.env,
            Path("outputs/operator_metrics") / args.env,
            Path("outputs/tables"),
            Path("outputs/figures"),
            (
                Path("outputs/benchmarks")
                / args.env
                / "lewm_regression_after_colab_gpu_sweep"
            ),
        ]
        for src in targets:
            if not src.exists() and not args.dry_run:
                continue
            dst = drive_root / src
            if args.dry_run:
                print(f"[dry-run] copy {src} -> {dst}")
            else:
                _copy_path(src, dst)
            copied_paths.append(str(dst))

        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None,
            "env_info": env_info,
            "args": vars(args),
            "paths_copied": copied_paths,
            "completed_stages": completed_stages,
            "failed_stages": failed_stages,
            "final_summary_table": (
                "outputs/tables/operator_split_summary_tworoom.csv"
            ),
        }
        manifest_path = drive_root / "RUN_MANIFEST.json"
        if args.dry_run:
            print(f"[dry-run] write manifest: {manifest_path}")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def stage_l() -> None:
        cuda_cmd = [
            "uv",
            "run",
            "python",
            "scripts/benchmark_cost_model.py",
            "--env",
            args.env,
            "--model-family",
            "lewm_hf",
            "--checkpoint",
            args.teacher_checkpoint,
            "--tag",
            "lewm_regression_after_colab_gpu_sweep",
            "--num-eval",
            "4",
            "--seed",
            "0",
            "--device",
            "cuda",
        ]
        ok, _ = _run_command(cuda_cmd, env=env, dry_run=args.dry_run)
        if ok:
            return
        print("[warn] CUDA regression guard failed; retrying on CPU.")
        cpu_cmd = cuda_cmd[:-1] + ["cpu"]
        ok_cpu, _ = _run_command(cpu_cmd, env=env, dry_run=args.dry_run)
        if not ok_cpu:
            raise RuntimeError("Regression guard failed on both CUDA and CPU.")

    stage_plan = [
        ("A", "env_check", stage_a),
        ("B", "build_caches", stage_b),
        ("C", "check_caches", stage_c),
        ("D", "compress_baselines", stage_d),
        ("E", "eval_baselines", stage_e),
        ("F", "distill_split", stage_f),
        ("G", "eval_distilled", stage_g),
        ("H", "summarize_split", stage_h),
        ("I", "plot_split", stage_i),
        ("J", "final_candidates", stage_j),
        ("K", "copy_to_drive", stage_k),
        ("L", "regression_guard", stage_l),
    ]

    for letter, key, fn in stage_plan:
        ok = run_stage(letter=letter, key=key, fn=fn)
        if not ok:
            break

    print("\n=== Sweep Summary ===")
    print(f"completed_stages: {completed_stages}")
    print(f"failed_stages: {failed_stages}")
    print(f"stage_timings_sec: {stage_timings}")

    if failed_stages and not args.continue_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
