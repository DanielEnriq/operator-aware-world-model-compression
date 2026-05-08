from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from oawc.compression import (
    count_parameters,
    factorize_linear_svd,
    model_size_bytes,
    relative_fro_error,
    save_json,
)
from oawc.compression.svd import factorized_linear_param_count
from oawc.models import load_cost_model


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable. "
            "Use --device cpu locally or run on Colab/H100."
        )
    return device_arg


def _get_child(parent: nn.Module, child_name: str) -> nn.Module:
    return getattr(parent, child_name)


def _set_child(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    setattr(parent, child_name, child)


def _parse_target_substrings(
    target_substrings: str | None,
    target_substring_legacy: str | None,
) -> list[str]:
    raw = target_substrings if target_substrings is not None else target_substring_legacy
    if raw is None:
        return ["predictor"]
    values = [v.strip() for v in str(raw).split(",") if v.strip()]
    return values or ["predictor"]


def _name_matches_any(name: str, substrings: list[str]) -> bool:
    return any(sub in name for sub in substrings)


def _params_matching_any_substring(
    model: nn.Module,
    substrings: list[str],
) -> int:
    total = 0
    for name, param in model.named_parameters():
        if _name_matches_any(name, substrings):
            total += int(param.numel())
    return total


def _infer_interface_mode(model: nn.Module) -> str:
    if callable(getattr(model, "get_cost", None)) or callable(
        getattr(model, "cost", None)
    ):
        return "cost_model_direct"
    if callable(getattr(model, "forward", None)) and not (
        callable(getattr(model, "encode", None))
        and callable(getattr(model, "rollout", None))
    ):
        return "forward_only"
    if callable(getattr(model, "encode", None)) and callable(
        getattr(model, "rollout", None)
    ):
        return "representation_rollout_only"
    return "no_planning_interface"


def replace_linear_by_path(
    model: nn.Module,
    full_name: str,
    replacement: nn.Module,
) -> None:
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    _set_child(parent, parts[-1], replacement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rank-fraction", type=float, required=True)
    parser.add_argument("--target-substring", default="predictor")
    parser.add_argument(
        "--target-substrings",
        default=None,
        help="Comma-separated module substrings to compress.",
    )
    parser.add_argument("--min-dim", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-root", default="outputs/compression")
    args = parser.parse_args()

    if not (0.0 < args.rank_fraction < 1.0):
        raise ValueError(
            "--rank-fraction must be in (0,1) for compressive SVD."
        )

    run_start = time.time()
    device = resolve_device(args.device)
    target_substrings = _parse_target_substrings(
        args.target_substrings,
        args.target_substring,
    )
    out_dir = Path(args.output_root) / args.env / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_cost_model(
        family=args.model_family,
        checkpoint=args.checkpoint,
        env_name=args.env,
        device=device,
    )
    model = loaded.model
    model = model.to(device).eval()
    model.requires_grad_(False)

    total_before = count_parameters(model)
    predictor_before = _params_matching_any_substring(model, target_substrings)
    size_before = model_size_bytes(model)

    linear_entries = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and _name_matches_any(name, target_substrings)
    ]

    module_report: list[dict] = []
    compressed_count = 0
    skipped_count = 0

    for name, layer in linear_entries:
        in_features = int(layer.in_features)
        out_features = int(layer.out_features)
        min_rank_dim = min(in_features, out_features)
        params_before = int(
            layer.weight.numel()
            + (layer.bias.numel() if layer.bias is not None else 0)
        )

        if min_rank_dim < args.min_dim:
            skipped_count += 1
            module_report.append(
                {
                    "name": name,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": None,
                    "params_before": params_before,
                    "params_after": params_before,
                    "compression_ratio": 1.0,
                    "relative_fro_error": 0.0,
                    "status": "skipped",
                    "skip_reason": f"min_dim<{args.min_dim}",
                }
            )
            continue

        rank = int(args.rank_fraction * min_rank_dim)
        rank = max(1, rank)
        if rank >= min_rank_dim:
            rank = min_rank_dim - 1

        if rank < 1:
            skipped_count += 1
            module_report.append(
                {
                    "name": name,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": rank,
                    "params_before": params_before,
                    "params_after": params_before,
                    "compression_ratio": 1.0,
                    "relative_fro_error": 0.0,
                    "status": "skipped",
                    "skip_reason": "rank_not_valid",
                }
            )
            continue

        params_after = factorized_linear_param_count(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            with_bias=layer.bias is not None,
        )
        if params_after >= params_before:
            skipped_count += 1
            module_report.append(
                {
                    "name": name,
                    "in_features": in_features,
                    "out_features": out_features,
                    "rank": rank,
                    "params_before": params_before,
                    "params_after": params_after,
                    "compression_ratio": float(params_after / params_before),
                    "relative_fro_error": 0.0,
                    "status": "skipped",
                    "skip_reason": "not_compressive",
                }
            )
            continue

        original_weight = layer.weight.detach().clone()
        factorized = factorize_linear_svd(layer, rank=rank)
        approx_weight = factorized.up.weight @ factorized.down.weight
        rel_error = relative_fro_error(original_weight, approx_weight)

        replace_linear_by_path(model, name, factorized)
        compressed_count += 1

        module_report.append(
            {
                "name": name,
                "in_features": in_features,
                "out_features": out_features,
                "rank": rank,
                "params_before": params_before,
                "params_after": params_after,
                "compression_ratio": float(params_after / params_before),
                "relative_fro_error": rel_error,
                "status": "compressed",
            }
        )

    skipped_count = int(len(linear_entries) - compressed_count)
    total_after = count_parameters(model)
    predictor_after = _params_matching_any_substring(model, target_substrings)
    size_after = model_size_bytes(model)

    compressed_model_path = out_dir / "compressed_model.pt"
    torch.save(model, compressed_model_path)

    compression_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "weight_svd",
        "env": args.env,
        "model_family": args.model_family,
        "checkpoint": args.checkpoint,
        "tag": args.tag,
        "target_substring": args.target_substring,
        "target_substrings": target_substrings,
        "min_dim": int(args.min_dim),
        "rank_fraction": float(args.rank_fraction),
        "device": device,
        "num_layers_considered": int(len(linear_entries)),
        "num_layers_compressed": int(compressed_count),
        "num_layers_skipped": int(skipped_count),
        "compression_status": (
            "no_op" if int(compressed_count) == 0 else "compressed"
        ),
        "total_params_before": int(total_before),
        "total_params_after": int(total_after),
        "predictor_params_before": int(predictor_before),
        "predictor_params_after": int(predictor_after),
        "total_compression_ratio": float(total_after / total_before),
        "predictor_compression_ratio": (
            float(predictor_after / predictor_before)
            if predictor_before > 0
            else None
        ),
        "model_size_bytes_before": int(size_before),
        "model_size_bytes_after": int(size_after),
        "has_get_cost_after_compression": bool(hasattr(model, "get_cost")),
        "interface_mode_after_compression": _infer_interface_mode(model),
        "selected_modules": [
            entry["name"]
            for entry in module_report
            if entry.get("status") == "compressed"
        ],
        "skipped_modules": [
            {
                "name": entry["name"],
                "reason_skipped": entry.get("skip_reason"),
            }
            for entry in module_report
            if entry.get("status") == "skipped"
        ],
        "layers_compressed": int(compressed_count),
        "wall_time_s": float(time.time() - run_start),
        "compressed_model_path": str(compressed_model_path),
    }
    save_json(out_dir / "compression_report.json", compression_report)
    save_json(out_dir / "module_report.json", {"layers": module_report})

    print("Weight-SVD predictor compression complete")
    print(f"  tag:                        {args.tag}")
    print(f"  layers considered:          {len(linear_entries)}")
    print(f"  layers compressed:          {compressed_count}")
    print(
        "  total compression ratio:    "
        f"{compression_report['total_compression_ratio']:.4f}"
    )
    print(
        "  predictor compression ratio:"
        f"{compression_report['predictor_compression_ratio']:.4f}"
        if compression_report["predictor_compression_ratio"] is not None
        else "  predictor compression ratio: n/a"
    )
    print(f"  compressed model:           {compressed_model_path}")
    print(
        "  report:                     "
        f"{out_dir / 'compression_report.json'}"
    )


if __name__ == "__main__":
    main()
