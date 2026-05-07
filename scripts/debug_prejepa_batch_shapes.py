from __future__ import annotations

import argparse
import importlib.util
from collections import OrderedDict
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import yaml
from omegaconf import OmegaConf, open_dict
from stable_worldmodel.wm.prejepa import module as prejepa_module
from torch import nn
from torch.utils.data import DataLoader


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_project_train_cfg(env_name: str) -> dict:
    cfg_path = (
        project_root() / "configs" / "train" / "swm_prejepa" / f"{env_name}.yaml"
    )
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_upstream_prejepa_module():
    path = (
        project_root()
        / "external"
        / "stable-worldmodel"
        / "scripts"
        / "train"
        / "prejepa.py"
    )
    spec = importlib.util.spec_from_file_location("upstream_prejepa_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_cfg(env_name: str, seed: int) -> tuple[object, str]:
    root = project_root()
    project_cfg = load_project_train_cfg(env_name)
    smoke = project_cfg["training"]["smoke"]
    out_name = f"oawc_swm_prejepa_{env_name}_seed{seed}_smoke"

    upstream_cfg_path = (
        root
        / "external"
        / "stable-worldmodel"
        / "scripts"
        / "train"
        / "config"
        / "prejepa.yaml"
    )
    cfg = OmegaConf.load(upstream_cfg_path)

    with open_dict(cfg):
        cfg.output_model_name = out_name
        cfg.subdir = out_name
        cfg.seed = seed
        cfg.cache_dir = str(root / ".swm_cache")
        cfg.dataset_name = project_cfg["dataset_name"]
        cfg.batch_size = smoke["batch_size"]
        cfg.num_workers = smoke["num_workers"]
        cfg.trainer.max_epochs = smoke["max_epochs"]
        cfg.trainer.accelerator = smoke["accelerator"]
        cfg.trainer.devices = smoke["devices"]
        cfg.trainer.precision = str(smoke["precision"])
        cfg.trainer.limit_train_batches = smoke["limit_train_batches"]
        cfg.trainer.limit_val_batches = smoke["limit_val_batches"]
        if "wandb" not in cfg:
            cfg.wandb = {"enabled": False, "config": {}}
        cfg.wandb.enabled = False

    return cfg, out_name


def normalize_sequence_tensor(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, 0.0)
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim == 1:
        return x.unsqueeze(0).unsqueeze(0)
    return x


def print_batch_info(batch: dict) -> None:
    for key in ("pixels", "action", "proprio", "id", "step_idx"):
        if key not in batch:
            print(f"{key}: <missing>")
            continue
        value = batch[key]
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key}: type={type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="tworoom", choices=["tworoom"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg, run_name = resolve_cfg(args.env, args.seed)
    upstream = load_upstream_prejepa_module()

    print("Resolved config values:")
    print(f"  dataset_name:   {cfg.dataset_name}")
    print(f"  n_steps:        {cfg.n_steps}")
    print(f"  frameskip:      {cfg.frameskip}")
    print(f"  wm.history_size:{cfg.wm.history_size}")
    print(f"  wm.num_preds:   {cfg.wm.num_preds}")
    print(f"  wm.encoding:    {dict(cfg.wm.get('encoding', {}))}")
    print(f"  run_name:       {run_name}")

    encoding_keys = list(cfg.wm.get("encoding", {}).keys())
    keys_to_load = ["pixels"] + encoding_keys

    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cfg.get("cache_dir", None),
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
    )
    normalizers = [
        upstream.get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get("encoding", {})
    ]
    transform = spt.data.transforms.Compose(
        upstream.get_img_preprocessor("pixels", "pixels", cfg.image_size),
        *normalizers,
    )
    dataset.transform = transform

    with open_dict(cfg):
        cfg.extra_dims = {}
        for key in cfg.wm.get("encoding", {}):
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = dim if key != "action" else dim * cfg.frameskip

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )

    batch = next(iter(loader))
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_batch = next(iter(val_loader))
    print("\nBatch shapes before forward:")
    print_batch_info(batch)
    print("\nValidation batch shapes:")
    print_batch_info(val_batch)

    for key in encoding_keys:
        if key in batch and torch.is_tensor(batch[key]):
            squeezed = torch.nan_to_num(batch[key], 0.0).squeeze()
            print(
                f"  upstream squeeze({key}) => shape={tuple(squeezed.shape)}"
            )

    encoder, embed_dim, num_patches, interp_pos_enc = upstream.get_encoder(cfg)
    embed_dim += sum(cfg.wm.get("encoding", {}).values())
    predictor_kwargs = {k: v for k, v in cfg.predictor.items() if k != "size"}

    predictor = prejepa_module.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.wm.history_size,
        dim=embed_dim,
        **predictor_kwargs,
    )

    extra_encoders = nn.ModuleDict(
        OrderedDict(
            (
                key,
                prejepa_module.Embedder(
                    in_chans=cfg.extra_dims[key], emb_dim=cfg.wm.encoding[key]
                ),
            )
            for key in encoding_keys
        )
    )
    for key, module in extra_encoders.items():
        module._oawc_debug_key = key  # type: ignore[attr-defined]

    world_model = swm.wm.PreJEPA(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.wm.history_size,
        num_pred=cfg.wm.num_preds,
        interpolate_pos_encoding=interp_pos_enc,
    ).eval()

    original_forward = prejepa_module.Embedder.forward

    def debug_forward(self, x):
        key = getattr(self, "_oawc_debug_key", "<unknown>")
        print(
            f"Embedder.forward key={key} input_shape={tuple(x.shape)} "
            f"dtype={x.dtype}"
        )
        return original_forward(self, x)

    prejepa_module.Embedder.forward = debug_forward
    try:
        buggy_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        for key in encoding_keys:
            buggy_batch[key] = torch.nan_to_num(buggy_batch[key], 0.0).squeeze()

        print("\nRunning encode with upstream squeeze behavior:")
        try:
            with torch.no_grad():
                world_model.encode(
                    buggy_batch,
                    target="emb",
                    is_video=cfg.backbone.get("is_video_encoder", False),
                )
            print("  encode succeeded unexpectedly with squeeze behavior")
        except Exception as exc:
            print(f"  encode failed: {type(exc).__name__}: {exc}")

        val_buggy_batch = {
            k: v.clone() if torch.is_tensor(v) else v for k, v in val_batch.items()
        }
        for key in encoding_keys:
            val_buggy_batch[key] = torch.nan_to_num(
                val_buggy_batch[key], 0.0
            ).squeeze()
            print(
                f"  val upstream squeeze({key}) => "
                f"shape={tuple(val_buggy_batch[key].shape)}"
            )
        print("\nRunning encode on validation batch with upstream squeeze behavior:")
        try:
            with torch.no_grad():
                world_model.encode(
                    val_buggy_batch,
                    target="emb",
                    is_video=cfg.backbone.get("is_video_encoder", False),
                )
            print("  val encode succeeded unexpectedly with squeeze behavior")
        except Exception as exc:
            print(f"  val encode failed: {type(exc).__name__}: {exc}")

        one_sample_buggy = {
            k: (v[:1].clone() if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        for key in encoding_keys:
            one_sample_buggy[key] = torch.nan_to_num(
                one_sample_buggy[key], 0.0
            ).squeeze()
            print(
                f"  one-sample upstream squeeze({key}) => "
                f"shape={tuple(one_sample_buggy[key].shape)}"
            )
        print("\nRunning encode on one-sample batch with upstream squeeze behavior:")
        try:
            with torch.no_grad():
                world_model.encode(
                    one_sample_buggy,
                    target="emb",
                    is_video=cfg.backbone.get("is_video_encoder", False),
                )
            print("  one-sample encode succeeded unexpectedly")
        except Exception as exc:
            print(f"  one-sample encode failed: {type(exc).__name__}: {exc}")

        safe_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
        for key in encoding_keys:
            safe_batch[key] = normalize_sequence_tensor(safe_batch[key])

        print("\nRunning encode with shape-safe behavior:")
        with torch.no_grad():
            encoded = world_model.encode(
                safe_batch,
                target="emb",
                is_video=cfg.backbone.get("is_video_encoder", False),
            )
        print(f"  safe encode succeeded; emb shape={tuple(encoded['emb'].shape)}")
    finally:
        prejepa_module.Embedder.forward = original_forward


if __name__ == "__main__":
    main()
