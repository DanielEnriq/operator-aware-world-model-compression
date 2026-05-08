from __future__ import annotations

import torch
from torch import nn


class FactorizedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        self.down = nn.Linear(
            in_features,
            rank,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up = nn.Linear(
            rank,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))
