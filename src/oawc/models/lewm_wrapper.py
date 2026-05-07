from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LeWMWrapper:
    """
    Thin wrapper around a loaded LeWM model.

    This is LeWM-specific. It exposes the operator-level methods that our
    compression/evaluation code will use later.
    """

    model: torch.nn.Module
    device: str | torch.device = "cpu"
    history_size: int = 3

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        pixels: (B, T, C, H, W), already preprocessed as LeWM expects.
        returns: emb, shape (B, T, D)
        """
        out = self.model.encode({"pixels": pixels.to(self.device)})
        return out["emb"]

    @torch.no_grad()
    def predict_from_emb(
        self,
        emb: torch.Tensor,
        action_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """
        emb: (B, H, D)
        action_blocks: (B, H, action_block * action_dim), already normalized.

        returns predicted embeddings, shape (B, H, D)
        """
        batch = {
            "pixels": torch.empty(0, device=self.device),  # unused placeholder
            "action": action_blocks.to(self.device),
        }

        act_emb = self.model.action_encoder(batch["action"])
        return self.model.predict(emb.to(self.device), act_emb)

    @torch.no_grad()
    def rollout(self, info: dict[str, Any], action_sequence: torch.Tensor) -> dict[str, Any]:
        """
        Calls LeWM's native rollout operator.

        This is the deployed autoregressive operator we eventually want to
        preserve under compression.
        """
        return self.model.rollout(
            info,
            action_sequence.to(self.device),
            history_size=self.history_size,
        )

    @torch.no_grad()
    def cost(self, info: dict[str, Any], action_candidates: torch.Tensor) -> torch.Tensor:
        """
        Calls LeWM's native planning-cost operator.
        """
        return self.model.get_cost(info, action_candidates.to(self.device))

    def compression_targets(self) -> dict[str, torch.nn.Module]:
        """
        Primary modules for operator-aware compression.

        The first target should be predictor only. Later experiments can include
        pred_proj and action_encoder.
        """
        return {
            "action_encoder": self.model.action_encoder,
            "predictor": self.model.predictor,
            "pred_proj": self.model.pred_proj,
        }