from __future__ import annotations

import importlib.util
from collections import OrderedDict
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from stable_worldmodel.wm.prejepa import module as prejepa_module
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from stable_worldmodel.wm.utils import save_pretrained


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_PREJEPA_PATH = (
    PROJECT_ROOT / "external" / "stable-worldmodel" / "scripts" / "train" / "prejepa.py"
)
UPSTREAM_CONFIG_DIR = (
    PROJECT_ROOT / "external" / "stable-worldmodel" / "scripts" / "train" / "config"
)


def _load_upstream_module():
    spec = importlib.util.spec_from_file_location(
        "upstream_prejepa_train", UPSTREAM_PREJEPA_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load upstream script: {UPSTREAM_PREJEPA_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPSTREAM = _load_upstream_module()


class SaveCkptCallback(Callback):
    def __init__(self, run_name, cfg, epoch_interval=1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
            save_pretrained(
                pl_module.model,
                run_name=self.run_name,
                config=self.cfg,
                filename=f"weights_epoch_{epoch}.pt",
            )


def _strip_action_dims(tensor, action_range):
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def _ensure_btd(x: torch.Tensor, key: str) -> torch.Tensor:
    x = torch.nan_to_num(x, 0.0)
    if x.ndim == 1:
        x = x.view(1, 1, -1)
    elif x.ndim == 2:
        x = x.unsqueeze(1)
    elif x.ndim == 4 and x.shape[2] == 1:
        x = x.squeeze(2)

    if x.ndim != 3:
        raise ValueError(f"Expected {key} to be [B, T, D], got shape={tuple(x.shape)}")
    return x


def dinowm_forward_compat(self, batch, stage, cfg):
    for key in self.model.extra_encoders:
        batch[key] = _ensure_btd(batch[key], key)

    batch = self.model.encode(
        batch,
        target="emb",
        is_video=cfg.backbone.get("is_video_encoder", False),
    )

    embedding = batch["emb"][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch["emb"][:, cfg.wm.num_preds :, ...].detach()

    pixels_dim = batch["pixels_emb"].size(-1)
    batch["pixels_loss"] = F.mse_loss(
        pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim]
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f"{key}_emb"].size(-1)
        lo, hi = start, start + dim
        if key == "action":
            action_range = [lo, hi]
        else:
            batch[f"{key}_loss"] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    batch["actionless_emb"] = _strip_action_dims(batch["emb"], action_range)
    batch["actionless_prev_emb"] = _strip_action_dims(embedding, action_range)
    batch["actionless_pred_emb"] = _strip_action_dims(pred_embedding, action_range)
    batch["actionless_target_emb"] = _strip_action_dims(
        target_embedding,
        action_range,
    )
    batch["loss"] = F.mse_loss(
        batch["actionless_pred_emb"],
        batch["actionless_target_emb"].detach(),
    )

    if batch["loss"].isnan():
        raise ValueError("NaN loss encountered!")

    self.log_dict(
        {f"{stage}/{k}": v.detach() for k, v in batch.items() if "_loss" in k},
        on_step=True,
        sync_dist=True,
    )
    return batch


def _estimate_tmax(cfg, train_loader_len: int) -> int:
    limit = cfg.trainer.get("limit_train_batches", None)
    if limit is None:
        per_epoch = train_loader_len
    elif isinstance(limit, float):
        per_epoch = max(1, int(train_loader_len * limit))
    else:
        per_epoch = max(1, int(limit))
    return max(1, int(cfg.trainer.max_epochs) * per_epoch)


@hydra.main(version_base=None, config_path=str(UPSTREAM_CONFIG_DIR), config_name="prejepa")
def run(cfg):
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
        UPSTREAM.get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get("encoding", {})
    ]
    if cfg.backbone.get("is_video_encoder", False):
        processor = UPSTREAM.AutoVideoProcessor.from_pretrained(cfg.backbone.name)
        transform = spt.data.transforms.Compose(
            UPSTREAM.VideoPipeline(processor, source="pixels", target="pixels"),
            spt.data.transforms.Resize(cfg.image_size, source="pixels", target="pixels"),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            UPSTREAM.get_img_preprocessor("pixels", "pixels", cfg.image_size),
            *normalizers,
        )
    dataset.transform = transform

    with open_dict(cfg):
        cfg.extra_dims = {}
        for key in cfg.wm.get("encoding", {}):
            if key not in dataset.column_names:
                raise ValueError(f"Encoding key '{key}' not found in dataset columns.")
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = dim if key != "action" else dim * cfg.frameskip
        if cfg.trainer.get("accelerator") == "cpu":
            cfg.trainer.strategy = "auto"

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        [cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True,
    )

    encoder, embed_dim, num_patches, interp_pos_enc = UPSTREAM.get_encoder(cfg)
    embed_dim += sum(cfg.wm.get("encoding", {}).values())
    if cfg.backbone.get("is_video_encoder", False):
        num_patches += num_patches * (cfg.n_steps // 4)

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
                    in_chans=cfg.extra_dims[key],
                    emb_dim=emb_dim,
                ),
            )
            for key, emb_dim in cfg.wm.get("encoding", {}).items()
        )
    )
    world_model = swm.wm.PreJEPA(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.wm.history_size,
        num_pred=cfg.wm.num_preds,
        interpolate_pos_encoding=interp_pos_enc,
    )
    t_max = _estimate_tmax(cfg, len(train_loader))
    module = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward_compat, cfg=cfg),
        optim={
            "model_opt": {
                "modules": "model",
                "optimizer": dict(cfg.optimizer),
                "scheduler": {"type": "CosineAnnealingLR", "T_max": t_max},
            }
        },
    )

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Compat PreJEPA run id: {run_id}")
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            SaveCkptCallback(run_name=cfg.output_model_name, cfg=cfg, epoch_interval=1),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    manager()
    object_ckpt = run_dir / f"{cfg.output_model_name}_object.ckpt"
    if object_ckpt.exists() or object_ckpt.is_symlink():
        object_ckpt.unlink()
    torch.save(module.model, object_ckpt)


if __name__ == "__main__":
    run()
