from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from oawc.benchmark import load_hdf5_dataset, sample_dataset_eval_tasks
from oawc.compression.activation_svd import (
    append_activation_rows,
    candidate_layer_rank,
    factorize_linear_activation_aware_with_fallback,
    is_compressive_linear,
    linear_param_count,
)
from oawc.compression.operator_metrics import (
    adapt_candidates_for_model,
    build_info_dict_from_cache,
    compute_model_costs,
    expand_info_for_candidates,
    maybe_align_action_width,
    resolve_device,
)
from oawc.compression.reports import (
    count_parameters,
    model_size_bytes,
    params_matching_substring,
    save_json,
)
from oawc.envs import ENV_SPECS
from oawc.models import load_cost_model


def _get_child(parent: nn.Module, child_name: str) -> nn.Module:
    return getattr(parent, child_name)


def _set_child(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    setattr(parent, child_name, child)


def replace_module_by_path(
    model: nn.Module,
    full_name: str,
    replacement: nn.Module,
) -> None:
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = _get_child(parent, part)
    _set_child(parent, parts[-1], replacement)


def _sample_candidates(
    *,
    num_states: int,
    num_candidates: int,
    horizon: int,
    action_dim: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    actions = (
        torch.rand(
            (num_states, num_candidates, horizon, action_dim),
            generator=gen,
            dtype=torch.float32,
        )
        * 2.0
        - 1.0
    )
    return actions.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rank-fraction", type=float, required=True)
    parser.add_argument("--target-substring", default="predictor")
    parser.add_argument("--min-dim", type=int, default=64)
    parser.add_argument("--num-calib-states", type=int, default=64)
    parser.add_argument("--num-calib-candidates", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-rows-per-layer", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--calib-source", default="random_candidates")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-root", default="outputs/compression")
    args = parser.parse_args()

    if args.model_family != "lewm_hf":
        raise ValueError(
            "Method 2 v0 currently supports --model-family lewm_hf."
        )
    if args.calib_source != "random_candidates":
        raise ValueError(
            "Method 2 v0 currently supports "
            "--calib-source random_candidates."
        )
    if not (0.0 < args.rank_fraction < 1.0):
        raise ValueError(
            "--rank-fraction must be in (0,1) for compressive SVD."
        )

    device = resolve_device(args.device)
    out_dir = Path(args.output_root) / args.env / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_cost_model(
        family=args.model_family,
        checkpoint=args.checkpoint,
        env_name=args.env,
        device=device,
    )
    model = loaded.model.to(device).eval()
    model.requires_grad_(False)

    total_before = count_parameters(model)
    predictor_before = params_matching_substring(model, args.target_substring)
    size_before = model_size_bytes(model)

    module_by_name = dict(model.named_modules())
    layer_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and args.target_substring in name
    ]

    # Collect calibration activations using forward pre-hooks.
    activation_rows: dict[str, torch.Tensor] = {}
    hooks = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed + 999)

    for name in layer_names:
        layer = module_by_name[name]
        if not isinstance(layer, nn.Linear):
            continue

        def _make_hook(layer_name):
            def _hook(_module, inputs):
                if not inputs:
                    return
                x = inputs[0]
                if not torch.is_tensor(x) or x.ndim < 2:
                    return
                rows = x.detach().reshape(-1, x.shape[-1]).float().cpu()
                activation_rows[layer_name] = append_activation_rows(
                    activation_rows.get(layer_name),
                    rows,
                    max_rows=args.max_rows_per_layer,
                    generator=gen,
                )

            return _hook

        hooks.append(layer.register_forward_pre_hook(_make_hook(name)))

    dataset = load_hdf5_dataset(args.env)
    tasks = sample_dataset_eval_tasks(
        dataset=dataset,
        goal_offset_steps=ENV_SPECS[args.env].goal_distance_steps,
        num_eval=args.num_calib_states,
        seed=args.seed,
    )
    info_dict = build_info_dict_from_cache(
        env_name=args.env,
        episodes_idx=list(tasks["episodes_idx"]),
        start_steps=list(tasks["start_steps"]),
        goal_offset_steps=int(tasks["goal_offset_steps"]),
        device=device,
    )
    info_dict = maybe_align_action_width(info_dict, model)
    action_dim = int(
        ENV_SPECS[args.env].action_dim or info_dict["action"].shape[-1]
    )
    candidate_actions = _sample_candidates(
        num_states=args.num_calib_states,
        num_candidates=args.num_calib_candidates,
        horizon=args.horizon,
        action_dim=action_dim,
        seed=args.seed,
        device=device,
    )
    candidate_actions_eval = adapt_candidates_for_model(candidate_actions, model)
    expanded_info = expand_info_for_candidates(
        info_dict,
        num_candidates=args.num_calib_candidates,
    )
    with torch.no_grad():
        _ = compute_model_costs(model, expanded_info, candidate_actions_eval)

    for h in hooks:
        h.remove()

    module_report: list[dict] = []
    num_compressed = 0
    num_fallback = 0
    weight_errors: list[float] = []
    activation_errors: list[float] = []

    for name in layer_names:
        layer = module_by_name[name]
        if not isinstance(layer, nn.Linear):
            continue

        in_features = int(layer.in_features)
        out_features = int(layer.out_features)
        params_before = linear_param_count(layer)
        min_rank_dim = min(in_features, out_features)
        rank = candidate_layer_rank(
            in_features=in_features,
            out_features=out_features,
            rank_fraction=args.rank_fraction,
        )
        entry = {
            "name": name,
            "in_features": in_features,
            "out_features": out_features,
            "rank": rank,
            "num_activation_rows": 0,
            "params_before": params_before,
            "params_after": params_before,
            "compression_ratio": 1.0,
            "relative_weight_error": None,
            "relative_activation_output_error": None,
            "ridge_used": None,
            "cholesky_retries": None,
            "status": "skipped",
            "skip_reason": None,
        }

        if min_rank_dim < args.min_dim:
            entry["skip_reason"] = f"min_dim<{args.min_dim}"
            module_report.append(entry)
            continue
        if rank < 1 or rank >= min_rank_dim:
            entry["skip_reason"] = "rank_not_valid"
            module_report.append(entry)
            continue
        if not is_compressive_linear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            has_bias=layer.bias is not None,
        ):
            entry["skip_reason"] = "not_compressive"
            module_report.append(entry)
            continue

        x_rows = activation_rows.get(name)
        if x_rows is None or x_rows.shape[0] == 0:
            entry["skip_reason"] = "no_calibration_activations"
            module_report.append(entry)
            continue

        x_rows = x_rows.to(device=device)
        if x_rows.shape[1] != in_features:
            entry["skip_reason"] = "activation_feature_mismatch"
            module_report.append(entry)
            continue

        result = factorize_linear_activation_aware_with_fallback(
            layer=layer,
            x_rows=x_rows,
            rank=rank,
            ridge=float(args.ridge),
        )
        replace_module_by_path(model, name, result.factorized)
        module_by_name[name] = result.factorized
        num_compressed += 1
        if result.used_fallback_weight_svd:
            num_fallback += 1
            status = "fallback_weight_svd"
        else:
            status = "compressed"

        params_after = int(
            result.factorized.down.weight.numel()
            + result.factorized.up.weight.numel()
            + (
                result.factorized.up.bias.numel()
                if result.factorized.up.bias is not None
                else 0
            )
        )
        entry.update(
            {
                "num_activation_rows": int(x_rows.shape[0]),
                "params_after": params_after,
                "compression_ratio": float(params_after / params_before),
                "relative_weight_error": result.relative_weight_error,
                "relative_activation_output_error": (
                    result.relative_activation_output_error
                ),
                "ridge_used": result.ridge_used,
                "cholesky_retries": result.cholesky_retries,
                "status": status,
                "skip_reason": None,
            }
        )
        module_report.append(entry)
        weight_errors.append(float(result.relative_weight_error))
        activation_errors.append(
            float(result.relative_activation_output_error)
        )

    total_after = count_parameters(model)
    predictor_after = params_matching_substring(model, args.target_substring)
    size_after = model_size_bytes(model)

    # Save model on CPU for portability.
    model = model.to("cpu")
    model.requires_grad_(False)
    compressed_model_path = out_dir / "compressed_model.pt"
    torch.save(model, compressed_model_path)

    num_skipped = len(layer_names) - num_compressed
    compression_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "activation_aware_svd",
        "env": args.env,
        "model_family": args.model_family,
        "checkpoint": args.checkpoint,
        "tag": args.tag,
        "rank_fraction": float(args.rank_fraction),
        "target_substring": args.target_substring,
        "min_dim": int(args.min_dim),
        "num_calib_states": int(args.num_calib_states),
        "num_calib_candidates": int(args.num_calib_candidates),
        "horizon": int(args.horizon),
        "max_rows_per_layer": int(args.max_rows_per_layer),
        "ridge": float(args.ridge),
        "calib_source": args.calib_source,
        "device": device,
        "num_layers_considered": int(len(layer_names)),
        "num_layers_compressed": int(num_compressed),
        "num_layers_skipped": int(num_skipped),
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
        "mean_relative_weight_error": (
            float(sum(weight_errors) / len(weight_errors))
            if weight_errors
            else None
        ),
        "mean_relative_activation_output_error": (
            float(sum(activation_errors) / len(activation_errors))
            if activation_errors
            else None
        ),
        "num_layers_fallback_to_weight_svd": int(num_fallback),
        "compressed_model_path": str(compressed_model_path),
    }

    save_json(out_dir / "compression_report.json", compression_report)
    save_json(out_dir / "module_report.json", {"layers": module_report})
    save_json(
        out_dir / "activation_report.json",
        {
            "layers": module_report,
            "max_rows_per_layer": int(args.max_rows_per_layer),
            "calib_source": args.calib_source,
        },
    )

    print("Activation-aware SVD predictor compression complete")
    print(f"  tag:                               {args.tag}")
    print(f"  layers considered:                 {len(layer_names)}")
    print(f"  layers compressed:                 {num_compressed}")
    print(f"  layers fallback to weight-SVD:     {num_fallback}")
    print(
        "  predictor compression ratio:       "
        f"{compression_report['predictor_compression_ratio']:.4f}"
    )
    print(
        "  total compression ratio:           "
        f"{compression_report['total_compression_ratio']:.4f}"
    )
    print(
        "  mean relative activation error:    "
        f"{compression_report['mean_relative_activation_output_error']:.6f}"
    )
    print(f"  compressed model:                  {compressed_model_path}")


if __name__ == "__main__":
    main()
