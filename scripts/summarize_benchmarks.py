from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def fmt_pct(x: Any) -> str:
    if x is None:
        return "—"
    return f"{float(x):.1f}%"


def fmt_num(x: Any) -> str:
    if x is None:
        return "—"
    return f"{float(x):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/benchmarks")
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("*/*/*.json"))

    if not files:
        raise SystemExit(f"No benchmark JSON files found under {root}")

    rows = []
    for path in files:
        data = load_json(path)

        env = data.get("environment", {}).get("name", path.parts[-3])
        model = data.get("model", {}).get("name", path.parts[-2])
        family = data.get("model", {}).get("family", "unknown")
        num_eval = data.get("evaluation_protocol", {}).get("num_eval")
        seed = data.get("evaluation_protocol", {}).get("seed")

        perf = data.get("performance", {})
        eff = data.get("efficiency", {})

        rows.append(
            {
                "env": env,
                "model": model,
                "family": family,
                "n": num_eval,
                "seed": seed,
                "success": perf.get("success_rate"),
                "time": eff.get("evaluation_time_sec"),
                "params": eff.get("model_parameters"),
                "path": str(path),
            }
        )

    rows.sort(key=lambda r: (r["env"], r["model"], r["n"] or 0, r["seed"] or 0))

    print("\nBenchmark summary")
    print("-" * 118)
    print(f"{'env':<14} {'model':<28} {'family':<14} {'n':>5} {'seed':>5} {'success':>9} {'time(s)':>10} {'params':>12}")
    print("-" * 118)

    for r in rows:
        params = "—" if r["params"] is None else f"{int(r['params'])/1e6:.2f}M"
        print(
            f"{r['env']:<14} "
            f"{r['model']:<28} "
            f"{r['family']:<14} "
            f"{str(r['n']):>5} "
            f"{str(r['seed']):>5} "
            f"{fmt_pct(r['success']):>9} "
            f"{fmt_num(r['time']):>10} "
            f"{params:>12}"
        )

    print("-" * 118)


if __name__ == "__main__":
    main()
