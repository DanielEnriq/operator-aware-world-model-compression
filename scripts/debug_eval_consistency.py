from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from oawc.compression.operator_eval import (
    evaluate_model_on_operator_cache,
    load_operator_cache,
)
from oawc.compression.operator_metrics import resolve_device
from oawc.compression.reports import save_json


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
        "old_value": float(a[i, j].item()),
        "crossed_value": float(b[i, j].item()),
        "abs_diff": float(diff[i, j].item()),
        "rel_diff": float(rel),
    }


def _agg(m: dict[str, Any]) -> dict[str, float]:
    return {
        "spearman": float(m["spearman_per_state"]["mean"]),
        "top5": float(m["topk_overlap"]["5"]["mean"]),
        "regret": float(m["teacher_regret"]["mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch-states", type=int, default=8)
    parser.add_argument("--batch-candidates", type=int, default=128)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    model_path = Path(args.model_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache: {cache_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    device = resolve_device(args.device)
    cache = load_operator_cache(cache_path)

    ref = evaluate_model_on_operator_cache(
        cache=cache,
        model_path=model_path,
        device=device,
        use_chunked_student=False,
    )
    crossed = evaluate_model_on_operator_cache(
        cache=cache,
        model_path=model_path,
        device=device,
        use_chunked_student=False,
    )
    probe = evaluate_model_on_operator_cache(
        cache=cache,
        model_path=model_path,
        device=device,
        use_chunked_student=True,
        batch_states=int(args.batch_states),
        batch_candidates=int(args.batch_candidates),
    )

    ref_costs = ref["student_costs"]
    crossed_costs = crossed["student_costs"]
    probe_costs = probe["student_costs"]
    exact_equal = bool(torch.equal(ref_costs, crossed_costs))
    allclose = bool(
        torch.allclose(ref_costs, crossed_costs, atol=float(args.atol), rtol=0.0)
    )
    max_abs = float((ref_costs - crossed_costs).abs().max().item())
    first = _first_divergence(ref_costs, crossed_costs, float(args.atol))
    probe_max_abs = float((ref_costs - probe_costs).abs().max().item())

    payload = {
        "cache_path": str(cache_path),
        "model_path": str(model_path),
        "device": device,
        "reference": {
            "metadata": ref["metadata"],
            "aggregate": _agg(ref["metrics"]),
            "teacher_costs_first3x8": [
                [float(v) for v in row]
                for row in ref["teacher_costs"][:3, :8].tolist()
            ],
            "student_costs_first3x8": [
                [float(v) for v in row]
                for row in ref_costs[:3, :8].tolist()
            ],
        },
        "crossed_exact": {
            "metadata": crossed["metadata"],
            "aggregate": _agg(crossed["metrics"]),
            "student_costs_first3x8": [
                [float(v) for v in row]
                for row in crossed_costs[:3, :8].tolist()
            ],
        },
        "chunked_probe": {
            "metadata": probe["metadata"],
            "max_abs_diff_vs_reference": probe_max_abs,
        },
        "comparison": {
            "exact_equal": exact_equal,
            "allclose": allclose,
            "max_abs_diff": max_abs,
            "first_divergence": first,
        },
    }

    out_dir = Path("outputs/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{model_path.parent.name}_{cache_path.parent.name}"
    out_path = out_dir / f"debug_eval_consistency_{tag}.json"
    save_json(out_path, payload)

    print("Debug eval consistency report written")
    print(f"  model_path: {model_path}")
    print(f"  cache_path: {cache_path}")
    print(f"  output:     {out_path}")
    print("  old aggregate:")
    print(
        "    spearman={:.6f} top5={:.6f} regret={:.6f}".format(
            payload["reference"]["aggregate"]["spearman"],
            payload["reference"]["aggregate"]["top5"],
            payload["reference"]["aggregate"]["regret"],
        )
    )
    print("  crossed aggregate:")
    print(
        "    spearman={:.6f} top5={:.6f} regret={:.6f}".format(
            payload["crossed_exact"]["aggregate"]["spearman"],
            payload["crossed_exact"]["aggregate"]["top5"],
            payload["crossed_exact"]["aggregate"]["regret"],
        )
    )
    print(
        "  comparison: exact_equal={} allclose={} max_abs_diff={:.8f}".format(
            exact_equal,
            allclose,
            max_abs,
        )
    )
    print(
        "  chunked probe max_abs_diff_vs_reference={:.8f}".format(
            probe_max_abs,
        )
    )
    if first is not None:
        print(
            (
                "  first divergence: state={} cand={} old={:.6f} "
                "crossed={:.6f} diff={:.6f}"
            ).format(
                first["state_index"],
                first["candidate_index"],
                first["old_value"],
                first["crossed_value"],
                first["abs_diff"],
            )
        )


if __name__ == "__main__":
    main()
