from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from oawc.benchmark import load_hdf5_dataset, sample_dataset_eval_tasks
from oawc.compression.operator_metrics import (
    adapt_candidates_for_model,
    build_info_dict_from_cache,
    expand_info_for_candidates,
    maybe_align_action_width,
)
from oawc.envs import ENV_SPECS


TARGET_KEYWORDS = [
    "predictor",
    "transition",
    "dynamics",
    "decoder",
    "encoder",
    "backbone",
    "projection",
    "projector",
    "mlp",
    "transformer",
    "block",
    "attn",
    "ff",
    "fc",
    "linear",
]


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return device_arg


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_parse_error": str(exc)}


def _read_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_parse_error": str(exc)}


def _short_repr(obj: Any, max_chars: int = 2400) -> str:
    text = repr(obj)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _is_tensor_dict(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    return all(torch.is_tensor(v) for v in payload.values())


def _is_state_dict(payload: dict[str, Any]) -> bool:
    if "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return True
    if not payload:
        return False
    return all(torch.is_tensor(v) for v in payload.values())


def _classify_loaded_object(loaded: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_type": f"{type(loaded).__module__}.{type(loaded).__name__}",
        "class_name": type(loaded).__name__,
        "classification": "unknown_object",
        "is_nn_module": isinstance(loaded, nn.Module),
        "is_dict_like": isinstance(loaded, dict),
        "top_level_keys": None,
    }
    if isinstance(loaded, nn.Module):
        info["classification"] = "full_nn_module"
        return info
    if isinstance(loaded, dict):
        info["top_level_keys"] = list(loaded.keys())[:200]
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            if "pytorch-lightning_version" in loaded:
                info["classification"] = "lightning_checkpoint"
            else:
                info["classification"] = "state_dict_container"
            return info
        if _is_tensor_dict(loaded):
            info["classification"] = "plain_tensor_dict"
            return info
        if _is_state_dict(loaded):
            info["classification"] = "state_dict"
            return info
        info["classification"] = "dict_unknown"
        return info
    return info


def _safe_signature(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except Exception:
        return None


def _module_param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _infer_interface_mode(model: nn.Module) -> str:
    has_get_cost = callable(getattr(model, "get_cost", None))
    has_cost = callable(getattr(model, "cost", None))
    has_forward = callable(getattr(model, "forward", None))
    has_encode = callable(getattr(model, "encode", None))
    has_rollout = callable(getattr(model, "rollout", None))
    if has_get_cost or has_cost:
        return "cost_model_direct"
    if has_forward and not (has_encode and has_rollout):
        return "forward_only"
    if has_encode and has_rollout:
        return "representation_rollout_only"
    if has_forward:
        return "forward_only"
    return "no_planning_interface"


def _recommend_target(name: str, module_type: str) -> tuple[bool, str]:
    lname = name.lower()
    if any(k in lname for k in ["predictor", "transition", "dynamics"]):
        return True, "predictor-like module"
    if any(k in lname for k in ["encoder", "backbone"]):
        return False, "encoder/backbone, avoid initially"
    if any(
        k in lname
        for k in [
            "projector",
            "projection",
            "mlp",
            "block",
            "attn",
            "ff",
            "fc",
            "linear",
            "decoder",
        ]
    ):
        return True, "secondary compression candidate"
    if module_type.startswith("Linear"):
        return True, "linear layer candidate"
    return False, "not a priority target"


def _epoch_num(path: Path) -> int:
    match = re.search(r"weights_epoch_(\d+)\.pt$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _prioritized_candidates(checkpoint_dir: Path) -> list[Path]:
    object_files = sorted(checkpoint_dir.glob("*_object.ckpt"))
    epoch_files = sorted(
        checkpoint_dir.glob("weights_epoch_*.pt"),
        key=_epoch_num,
        reverse=True,
    )
    weights_ckpt_files = sorted(checkpoint_dir.glob("*_weights.ckpt"))
    others = sorted(
        p
        for p in checkpoint_dir.iterdir()
        if p.is_file() and p.suffix in {".pt", ".ckpt"}
    )
    ordered = object_files + epoch_files + weights_ckpt_files
    for p in others:
        if p not in ordered:
            ordered.append(p)
    return ordered


def _top_level_group(name: str) -> str:
    if "." not in name:
        return name or "__root__"
    return name.split(".", 1)[0]


def _build_probe_inputs(
    *,
    env_name: str,
    model: nn.Module,
    max_probe_batch: int,
    device: str,
) -> dict[str, Any]:
    dataset = load_hdf5_dataset(env_name)
    spec = ENV_SPECS[env_name]
    tasks = sample_dataset_eval_tasks(
        dataset=dataset,
        goal_offset_steps=spec.goal_distance_steps,
        num_eval=max_probe_batch,
        seed=0,
    )
    info_dict = build_info_dict_from_cache(
        env_name=env_name,
        episodes_idx=list(tasks["episodes_idx"]),
        start_steps=list(tasks["start_steps"]),
        goal_offset_steps=int(tasks["goal_offset_steps"]),
        device="cpu",
    )
    info_dict = maybe_align_action_width(info_dict, model)
    action_dim = int(spec.action_dim or info_dict["action"].shape[-1])
    num_states = int(info_dict["action"].shape[0])
    num_candidates = 4
    horizon = 2
    candidates = (
        torch.rand(
            (num_states, num_candidates, horizon, action_dim),
            dtype=torch.float32,
        )
        * 2.0
        - 1.0
    )
    candidate_eval = adapt_candidates_for_model(candidates, model)
    expanded_info = expand_info_for_candidates(
        info_dict,
        num_candidates=num_candidates,
    )
    expanded_info = {
        k: v.to(device) for k, v in expanded_info.items() if torch.is_tensor(v)
    }
    candidate_eval = candidate_eval.to(device)
    return {
        "expanded_info": expanded_info,
        "candidate_eval": candidate_eval,
        "num_states": num_states,
        "num_candidates": num_candidates,
        "horizon": horizon,
        "action_dim": int(candidate_eval.shape[-1]),
    }


def _probe_interface_calls(
    *,
    model: nn.Module,
    interface_mode: str,
    env_name: str,
    max_probe_batch: int,
    device: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "attempted": False,
        "status": "not_attempted",
        "details": {},
    }
    if interface_mode == "no_planning_interface":
        out["status"] = "skipped_no_interface"
        return out
    try:
        probe_inputs = _build_probe_inputs(
            env_name=env_name,
            model=model,
            max_probe_batch=max_probe_batch,
            device=device,
        )
    except Exception as exc:
        out["status"] = "probe_input_failed"
        out["details"]["error"] = str(exc)
        return out
    info_dict = probe_inputs["expanded_info"]
    action_candidates = probe_inputs["candidate_eval"]
    out["attempted"] = True
    out["details"]["probe_input"] = {
        k: list(v.shape) for k, v in info_dict.items()
    }
    out["details"]["candidate_shape"] = list(action_candidates.shape)
    with torch.no_grad():
        if interface_mode == "cost_model_direct":
            fn = (
                getattr(model, "get_cost", None)
                if callable(getattr(model, "get_cost", None))
                else getattr(model, "cost", None)
            )
            if not callable(fn):
                out["status"] = "failed_no_callable"
                return out
            try:
                result = fn(info_dict, action_candidates)
                out["status"] = "ok"
                out["details"]["result_shape"] = list(result.shape)
            except Exception as exc:
                out["status"] = "call_failed"
                out["details"]["error"] = str(exc)
            return out
        if interface_mode == "forward_only":
            fn = getattr(model, "forward")
            try:
                result = fn(info_dict, action_candidates)
                if torch.is_tensor(result):
                    out["details"]["result_shape"] = list(result.shape)
                else:
                    out["details"]["result_type"] = str(type(result))
                out["status"] = "ok"
            except Exception as exc:
                out["status"] = "call_failed"
                out["details"]["error"] = str(exc)
            return out
        if interface_mode == "representation_rollout_only":
            encode_fn = getattr(model, "encode", None)
            rollout_fn = getattr(model, "rollout", None)
            if not callable(encode_fn) or not callable(rollout_fn):
                out["status"] = "failed_no_callable"
                return out
            try:
                goal = {
                    k: v[:, 0]
                    for k, v in info_dict.items()
                    if torch.is_tensor(v)
                }
                if "goal" in goal and "pixels" not in goal:
                    goal["pixels"] = goal["goal"]
                for key in list(goal.keys()):
                    if key.startswith("goal_"):
                        goal[key[len("goal_"):]] = goal.pop(key)
                goal.pop("action", None)
                encode_out = encode_fn(goal)
                if isinstance(encode_out, dict) and "emb" in encode_out:
                    rollout_info = dict(info_dict)
                    rollout_info["goal_emb"] = encode_out["emb"]
                    rollout_out = rollout_fn(rollout_info, action_candidates)
                    out["details"]["encode_keys"] = list(
                        encode_out.keys()
                    )[:50]
                    out["details"]["rollout_type"] = str(type(rollout_out))
                else:
                    out["details"]["encode_type"] = str(type(encode_out))
                out["status"] = "ok"
            except Exception as exc:
                out["status"] = "call_failed"
                out["details"]["error"] = str(exc)
            return out
    return out


def _summarize_module(model: nn.Module) -> dict[str, Any]:
    total_params = _module_param_count(model)
    trainable_params = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    children = [
        {
            "name": name,
            "type": type(child).__name__,
            "params": _module_param_count(child),
        }
        for name, child in model.named_children()
    ]
    module_rows = []
    linear_conv_rows = []
    per_top_level: dict[str, int] = {}
    for name, param in model.named_parameters():
        per_top_level.setdefault(_top_level_group(name), 0)
        per_top_level[_top_level_group(name)] += int(param.numel())
    for name, module in model.named_modules():
        mtype = type(module).__name__
        if name:
            module_rows.append({"name": name, "type": mtype})
        if isinstance(module, nn.Linear):
            linear_conv_rows.append(
                {
                    "name": name,
                    "type": "Linear",
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "params": _module_param_count(module),
                    "top_level_group": _top_level_group(name),
                }
            )
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            linear_conv_rows.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "in_channels": int(module.in_channels),
                    "out_channels": int(module.out_channels),
                    "kernel_size": str(module.kernel_size),
                    "params": _module_param_count(module),
                    "top_level_group": _top_level_group(name),
                }
            )
    return {
        "repr_summary": _short_repr(model),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "named_children": children,
        "named_modules_count": len(module_rows),
        "named_modules_sample": module_rows[:250],
        "named_parameters_top_level": per_top_level,
        "leaf_linear_conv_layers": linear_conv_rows,
    }


def _build_compression_targets(model: nn.Module) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        lname = name.lower()
        if not any(k in lname for k in TARGET_KEYWORDS):
            continue
        mtype = type(module).__name__
        params = _module_param_count(module)
        top_group = _top_level_group(name)
        recommended, reason = _recommend_target(name, mtype)
        row: dict[str, Any] = {
            "name": name,
            "type": mtype,
            "params": params,
            "top_level_group": top_group,
            "recommended": recommended,
            "reason": reason,
        }
        if isinstance(module, nn.Linear):
            row["in_features"] = int(module.in_features)
            row["out_features"] = int(module.out_features)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            row["in_channels"] = int(module.in_channels)
            row["out_channels"] = int(module.out_channels)
            row["kernel_size"] = str(module.kernel_size)
        targets.append(row)
    targets.sort(
        key=lambda r: (
            0 if r["recommended"] else 1,
            str(r["top_level_group"]),
            -int(r["params"]),
        )
    )
    return targets


def _recommended_param_total(
    model: nn.Module,
    substrings: list[str],
) -> int:
    wanted = [s.lower() for s in substrings]
    total = 0
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(s in lname for s in wanted):
            if "encoder" in lname or "backbone" in lname:
                continue
            total += int(param.numel())
    return total


def _artifact_group(path: Path) -> str:
    name = path.name
    if name.endswith("_object.ckpt"):
        return "object_ckpt"
    if re.search(r"weights_epoch_\d+\.pt$", name):
        return "weights_epoch_pt"
    if name.endswith("_weights.ckpt"):
        return "weights_ckpt"
    return "other_pt_or_ckpt"


def _pick_recommended_artifact(
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    priority = {
        "object_ckpt": 0,
        "weights_epoch_pt": 1,
        "weights_ckpt": 2,
        "other_pt_or_ckpt": 3,
    }
    ordered = sorted(
        audit_rows,
        key=lambda r: (
            priority.get(str(r.get("artifact_group")), 9),
            -int(r.get("epoch", -1)),
            str(r["path"]),
        ),
    )
    if not ordered:
        return {
            "path": None,
            "reason": "no_artifacts_found",
            "loadable": False,
        }
    for row in ordered:
        if row.get("load_status") == "ok" and row.get("is_module"):
            return {
                "path": row["path"],
                "reason": "highest-priority loadable module",
                "loadable": True,
                "artifact_group": row.get("artifact_group"),
            }
    first = ordered[0]
    return {
        "path": first["path"],
        "reason": "priority artifact exists but not loadable module",
        "loadable": False,
        "artifact_group": first.get("artifact_group"),
    }


def _load_artifact(
    path: Path,
    *,
    device: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(path),
        "artifact_group": _artifact_group(path),
        "epoch": _epoch_num(path),
        "load_status": "failed",
        "error": None,
        "classification": None,
        "python_type": None,
        "class_name": None,
        "top_level_keys": None,
        "is_module": False,
        "module_summary": None,
        "interface": None,
    }
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        row["error"] = str(exc)
        return row
    info = _classify_loaded_object(loaded)
    row.update(
        {
            "load_status": "ok",
            "classification": info["classification"],
            "python_type": info["python_type"],
            "class_name": info["class_name"],
            "top_level_keys": info["top_level_keys"],
            "is_module": bool(info["is_nn_module"]),
        }
    )
    if isinstance(loaded, nn.Module):
        row["module_summary"] = _summarize_module(loaded)
        interface_methods = {}
        for name in [
            "get_cost",
            "cost",
            "forward",
            "encode",
            "rollout",
            "predict",
            "step",
            "encoder",
            "predictor",
            "transition",
            "dynamics",
            "decoder",
        ]:
            attr = getattr(loaded, name, None)
            interface_methods[name] = {
                "exists": hasattr(loaded, name),
                "callable": callable(attr),
                "signature": _safe_signature(attr) if callable(attr) else None,
            }
        row["interface"] = {
            "methods": interface_methods,
            "interface_mode": _infer_interface_mode(loaded),
        }
    return row


def _format_md(report: dict[str, Any]) -> str:
    lines = [
        f"# SWM PreJEPA Checkpoint Audit ({report['env']})",
        "",
        f"- checkpoint_dir: `{report['checkpoint_dir']}`",
        f"- recommended_artifact: "
        f"`{report['recommended_artifact'].get('path')}`",
        f"- recommended_loadable: "
        f"{report['recommended_artifact'].get('loadable')}",
        "",
        "## Files found",
    ]
    for item in report["files_found"]:
        lines.append(f"- `{item['name']}` ({item['size_bytes']} bytes)")
    lines.extend(["", "## Artifact load summary"])
    for row in report["artifacts"]:
        lines.append(
            "- "
            f"`{Path(row['path']).name}` "
            f"[{row.get('artifact_group')}] "
            f"status={row['load_status']} "
            f"classification={row.get('classification')} "
            f"interface_mode="
            f"{(row.get('interface') or {}).get('interface_mode')}"
        )
        if row.get("error"):
            lines.append(f"  - error: `{row['error']}`")
    if report.get("selected_model") is not None:
        sel = report["selected_model"]
        lines.extend(
            [
                "",
                "## Selected model",
                f"- source: `{sel['source_path']}`",
                f"- interface_mode: `{sel['interface_mode']}`",
                f"- total_params: {sel['total_params']}",
                f"- trainable_params: {sel['trainable_params']}",
                f"- closed_loop_readiness: `{sel['closed_loop_readiness']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Compression target recommendation",
            f"- recommended_substrings: "
            f"`{', '.join(report['target_recommendation']['substrings'])}`",
            f"- target_params: "
            f"{report['target_recommendation']['target_params']}",
            f"- total_params: "
            f"{report['target_recommendation']['total_params']}",
            f"- note: {report['target_recommendation']['note']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/audits/swm_prejepa",
    )
    parser.add_argument("--probe-interface", action="store_true")
    parser.add_argument("--max-probe-batch", type=int, default=2)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_found = sorted(
        [
            {
                "name": p.name,
                "size_bytes": int(p.stat().st_size),
                "suffix": p.suffix,
            }
            for p in checkpoint_dir.iterdir()
            if p.is_file()
        ],
        key=lambda x: x["name"],
    )

    candidates = _prioritized_candidates(checkpoint_dir)
    artifact_rows = [
        _load_artifact(path, device=device) for path in candidates
    ]
    recommended_artifact = _pick_recommended_artifact(artifact_rows)

    selected_row = None
    selected_model: nn.Module | None = None
    if recommended_artifact["path"] is not None:
        selected_row = next(
            (
                row
                for row in artifact_rows
                if row["path"] == recommended_artifact["path"]
            ),
            None,
        )
    if selected_row and selected_row.get("is_module"):
        selected_model = torch.load(
            selected_row["path"],
            map_location=device,
            weights_only=False,
        )

    selected_model_report = None
    targets: list[dict[str, Any]] = []
    target_recommendation = {
        "substrings": [],
        "target_params": 0,
        "total_params": 0,
        "note": "no_module_loaded",
    }
    if isinstance(selected_model, nn.Module):
        module_summary = _summarize_module(selected_model)
        interface_mode = _infer_interface_mode(selected_model)
        selected_model_report = {
            "source_path": str(selected_row["path"]) if selected_row else None,
            "interface_mode": interface_mode,
            "total_params": module_summary["total_params"],
            "trainable_params": module_summary["trainable_params"],
            "closed_loop_readiness": (
                "direct_cost_interface"
                if interface_mode == "cost_model_direct"
                else (
                    "benchmark_forward_alias_possible"
                    if interface_mode == "forward_only"
                    else (
                        "representation_rollout_fallback_only"
                        if interface_mode == "representation_rollout_only"
                        else "no_planning_interface"
                    )
                )
            ),
            "probe": None,
        }
        if args.probe_interface:
            selected_model = selected_model.to(device).eval()
            selected_model.requires_grad_(False)
            selected_model_report["probe"] = _probe_interface_calls(
                model=selected_model,
                interface_mode=interface_mode,
                env_name=args.env,
                max_probe_batch=max(1, int(args.max_probe_batch)),
                device=device,
            )
        targets = _build_compression_targets(selected_model)
        preferred_substrings = [
            "predictor",
            "transition",
            "dynamics",
            "projector",
            "mlp",
        ]
        rec_target_params = _recommended_param_total(
            selected_model,
            preferred_substrings,
        )
        target_recommendation = {
            "substrings": preferred_substrings,
            "target_params": rec_target_params,
            "total_params": int(module_summary["total_params"]),
            "note": (
                "Prioritize predictor/transition/dynamics; "
                "projector/mlp as secondary; avoid encoder/backbone initially."
            ),
        }

    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "env": args.env,
        "device": device,
        "files_found": files_found,
        "config_yaml": _read_yaml(checkpoint_dir / "config.yaml"),
        "config_json": _read_json(checkpoint_dir / "config.json"),
        "artifacts": artifact_rows,
        "recommended_artifact": recommended_artifact,
        "selected_model": selected_model_report,
        "target_recommendation": target_recommendation,
    }

    json_path = output_dir / f"{args.env}_checkpoint_audit.json"
    md_path = output_dir / f"{args.env}_checkpoint_audit.md"
    targets_path = output_dir / f"{args.env}_compression_targets.json"

    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_md(report), encoding="utf-8")
    targets_path.write_text(
        json.dumps({"env": args.env, "targets": targets}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[done] wrote {json_path}")
    print(f"[done] wrote {md_path}")
    print(f"[done] wrote {targets_path}")


if __name__ == "__main__":
    main()
