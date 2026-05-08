from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oawc.compression.operator_eval import rank_fraction_to_tag


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


def _parse_ranks(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="quentinll/lewm-tworooms",
    )
    parser.add_argument("--num-eval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-teacher", action="store_true", default=True)
    parser.add_argument(
        "--ranks",
        default="0.95,0.90,0.80,0.70,0.60,0.50",
    )
    parser.add_argument("--methods", default="svd,aa_svd")
    args = parser.parse_args()

    ranks = _parse_ranks(args.ranks)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    env_vars = os.environ.copy()
    env_vars.setdefault("PYTHONPATH", str(Path.cwd() / "src"))

    rows: list[dict[str, Any]] = []
    if args.include_teacher:
        tag = (
            "lewm_tworoom_teacher_closed_loop_"
            f"seed{args.seed}_n{args.num_eval}"
        )
        out_path = (
            Path("outputs/benchmarks")
            / args.env
            / tag
            / f"{tag}_seed{args.seed}_n{args.num_eval}.json"
        )
        if not (args.skip_existing and out_path.exists()):
            ok, elapsed = _run(
                [
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
                    tag,
                    "--num-eval",
                    str(args.num_eval),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                ],
                env=env_vars,
                dry_run=args.dry_run,
            )
            rows.append(
                {
                    "tag": tag,
                    "kind": "teacher",
                    "ok": ok,
                    "elapsed_sec": elapsed,
                }
            )

    for method in methods:
        for rank in ranks:
            rtag = rank_fraction_to_tag(rank)
            if method == "svd":
                model_tag = f"lewm_tworoom_svd_{rtag}"
            elif method == "aa_svd":
                model_tag = f"lewm_tworoom_aa_svd_{rtag}"
            else:
                raise ValueError(f"Unknown method: {method}")
            model_path = (
                Path("outputs/compression")
                / args.env
                / model_tag
                / "compressed_model.pt"
            )
            if not model_path.exists() and not args.dry_run:
                rows.append(
                    {
                        "tag": model_tag,
                        "kind": "compressed",
                        "status": "missing_model",
                        "model_path": str(model_path),
                    }
                )
                continue
            tag = (
                f"{model_tag}_closed_loop_seed{args.seed}_n{args.num_eval}"
            )
            out_path = (
                Path("outputs/benchmarks")
                / args.env
                / tag
                / f"{tag}_seed{args.seed}_n{args.num_eval}.json"
            )
            if args.skip_existing and out_path.exists():
                rows.append(
                    {
                        "tag": tag,
                        "kind": "compressed",
                        "status": "skipped_existing",
                    }
                )
                continue
            ok, elapsed = _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/benchmark_cost_model.py",
                    "--env",
                    args.env,
                    "--model-path",
                    str(model_path),
                    "--tag",
                    tag,
                    "--num-eval",
                    str(args.num_eval),
                    "--seed",
                    str(args.seed),
                    "--device",
                    args.device,
                ],
                env=env_vars,
                dry_run=args.dry_run,
            )
            rows.append(
                {
                    "tag": tag,
                    "model_tag": model_tag,
                    "kind": "compressed",
                    "ok": ok,
                    "elapsed_sec": elapsed,
                    "model_path": str(model_path),
                }
            )

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "env": args.env,
        "num_eval": int(args.num_eval),
        "seed": int(args.seed),
        "rows": rows,
    }
    out_path = Path(
        "outputs/tables/closed_loop_rank_frontier_manifest_tworoom.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote manifest: {out_path}")


if __name__ == "__main__":
    main()
