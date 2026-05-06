from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from transformers import ViTConfig, ViTModel


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEWM_SRC = PROJECT_ROOT / "external" / "le-wm"


def _ensure_lewm_source() -> None:
    if not LEWM_SRC.exists():
        raise FileNotFoundError(
            f"Missing LeWM source at {LEWM_SRC}. "
            "Run: git clone https://github.com/lucas-maes/le-wm.git external/le-wm"
        )

    if str(LEWM_SRC) not in sys.path:
        sys.path.insert(0, str(LEWM_SRC))


def _filter_hydra(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _vit_hf(
    size: str,
    patch_size: int,
    image_size: int,
    pretrained: bool = False,
    use_mask_token: bool = False,
):
    if size != "tiny":
        raise ValueError(f"Only LeWM tiny config is supported for now, got size={size!r}")

    if pretrained:
        raise ValueError("This loader currently expects non-pretrained LeWM ViT configs.")

    config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
    )

    return ViTModel(config, add_pooling_layer=False)


def load_lewm_from_hf(
    checkpoint_repo: str,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    _ensure_lewm_source()

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    weights_path = hf_hub_download(repo_id=checkpoint_repo, filename="weights.pt")
    config_path = hf_hub_download(repo_id=checkpoint_repo, filename="config.json")

    with open(config_path) as f:
        cfg = json.load(f)

    encoder = _vit_hf(
        size=cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=cfg["encoder"].get("pretrained", False),
        use_mask_token=cfg["encoder"].get("use_mask_token", False),
    )

    def build_mlp(key: str) -> torch.nn.Module:
        return MLP(
            input_dim=cfg[key]["input_dim"],
            output_dim=cfg[key]["output_dim"],
            hidden_dim=cfg[key]["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**_filter_hydra(cfg["predictor"])),
        action_encoder=Embedder(**_filter_hydra(cfg["action_encoder"])),
        projector=build_mlp("projector"),
        pred_proj=build_mlp("pred_proj"),
    )

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    model = model.to(device)
    model.eval()

    return model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())