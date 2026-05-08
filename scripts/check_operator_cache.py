from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _stats(x: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(x.min()),
        "mean": float(x.mean()),
        "max": float(x.max()),
        "std": float(x.std()),
    }


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_path")
    parser.add_argument("--summary-json", action="store_true")
    args = parser.parse_args()

    cache_path = Path(args.cache_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    costs = cache["teacher_costs"]
    candidates = cache["candidate_actions"]
    best_idx = cache["teacher_best_index"]
    best_first = cache["teacher_best_first_action"]
    topk_indices = cache.get("topk_indices", {})
    n_states, n_candidates = costs.shape

    print("=== Operator Cache Check ===")
    print(f"path: {cache_path}")
    print(f"env: {cache.get('env')}")
    print(f"model_family: {cache.get('model_family')}")
    print(f"checkpoint: {cache.get('checkpoint')}")
    print(f"tag: {cache.get('tag')}")
    print()
    print(f"candidate_actions shape: {tuple(candidates.shape)}")
    print(f"teacher_costs shape: {tuple(costs.shape)}")
    print(f"teacher_best_index shape: {tuple(best_idx.shape)}")
    print(f"teacher_best_first_action shape: {tuple(best_first.shape)}")
    print()

    finite = bool(torch.isfinite(costs).all().item())
    global_stats = _stats(costs)
    per_state_std = costs.std(dim=1)
    per_state_std_stats = _stats(per_state_std)
    near_constant_mask = per_state_std < 1e-6
    near_constant_count = int(near_constant_mask.sum().item())
    unique_best_index_count = int(torch.unique(best_idx).numel())

    warnings: list[str] = []
    if not finite:
        warnings.append("Non-finite teacher costs detected.")
    if near_constant_count > 0:
        warnings.append(
            f"Near-constant cost states detected: {near_constant_count} "
            "(per-state std < 1e-6)."
        )

    print(f"costs finite: {bool(finite)}")
    print(
        "global cost stats: "
        f"min={global_stats['min']:.6f}, "
        f"mean={global_stats['mean']:.6f}, "
        f"max={global_stats['max']:.6f}, "
        f"std={global_stats['std']:.6f}"
    )
    print(
        "per-state cost std stats: "
        f"min={per_state_std_stats['min']:.6f}, "
        f"mean={per_state_std_stats['mean']:.6f}, "
        f"max={per_state_std_stats['max']:.6f}"
    )
    print(f"near-constant states (std < 1e-6): {near_constant_count}/{n_states}")
    print(f"unique teacher_best_index values: {unique_best_index_count}")
    print()

    # Top-k checks
    sorted_idx = torch.argsort(costs, dim=1)
    topk_checks: dict[str, dict[str, bool]] = {}
    print("top-k shape checks:")
    for key, idx in sorted(topk_indices.items(), key=lambda kv: int(kv[0])):
        k = int(key)
        expected = (n_states, k)
        ok_shape = tuple(idx.shape) == expected
        ok_range = (
            idx.dtype in (torch.int64, torch.int32)
            and int(idx.min()) >= 0
            and int(idx.max()) < n_candidates
        )
        ok_sorted = torch.equal(idx, sorted_idx[:, :k])
        topk_checks[key] = {
            "shape_ok": bool(ok_shape),
            "range_ok": bool(ok_range),
            "exact_rank_ok": bool(ok_sorted),
        }
        print(
            f"  k={k}: shape_ok={ok_shape}, "
            f"range_ok={bool(ok_range)}, exact_rank_ok={bool(ok_sorted)}"
        )
    print()

    best_match = torch.equal(best_idx, sorted_idx[:, 0])
    if not best_match:
        warnings.append("teacher_best_index does not match argmin(cost).")
    print(f"best-index matches argmin(cost): {bool(best_match)}")
    print(f"first best indices: {best_idx[: min(8, len(best_idx))].tolist()}")
    print("first best first-actions:")
    for row in best_first[: min(5, len(best_first))]:
        print(f"  {row.tolist()}")

    topk_nesting_checks: dict[str, bool] = {}
    sorted_k = sorted(int(k) for k in topk_indices.keys())
    if sorted_k:
        print()
        print("top-k nesting checks:")
    for left, right in zip(sorted_k, sorted_k[1:]):
        left_key = str(left)
        right_key = str(right)
        left_vals = topk_indices[left_key]
        right_vals = topk_indices[right_key]
        nesting_ok = True
        for i in range(n_states):
            left_set = set(left_vals[i].tolist())
            right_set = set(right_vals[i].tolist())
            if not left_set.issubset(right_set):
                nesting_ok = False
                break
        check_key = f"{left}_subset_{right}"
        topk_nesting_checks[check_key] = nesting_ok
        print(f"  {left} subset {right}: {nesting_ok}")
        if not nesting_ok:
            warnings.append(f"Top-k nesting failed: {left} subset {right}.")

    if warnings:
        print()
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if args.summary_json:
        summary = {
            "cache_path": str(cache_path),
            "finite_costs": finite,
            "global_cost_stats": global_stats,
            "per_state_cost_std_stats": per_state_std_stats,
            "near_constant_state_count": near_constant_count,
            "unique_best_index_count": unique_best_index_count,
            "argmin_matches": bool(best_match),
            "topk_checks": topk_checks,
            "topk_nesting_checks": topk_nesting_checks,
            "warnings": warnings,
        }
        out_path = cache_path.parent / "cache_diagnostics.json"
        out_path.write_text(json.dumps(_to_jsonable(summary), indent=2))
        print()
        print(f"summary json saved: {out_path}")


if __name__ == "__main__":
    main()
