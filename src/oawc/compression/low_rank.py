from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import Iterable

import torch
from torch import nn


@dataclass
class LowRankReport:
    method: str
    target: str
    rank_fraction: float
    min_rank: int
    num_linear_replaced: int
    original_target_params: int
    compressed_target_params: int
    target_param_ratio: float
    target_param_reduction: float

    def to_dict(self) -> dict:
        return asdict(self)


class LowRankLinear(nn.Module):
    """
    Dense low-rank replacement for nn.Linear.

    Original:
        y = x W^T + b

    Low-rank:
        W ≈ A B
        where B: rank x in_features
              A: out_features x rank

        y = A(Bx) + b

    Implemented as:
        down: in_features -> rank, no bias
        up: rank -> out_features, optional original bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

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


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def svd_compress_linear(
    linear: nn.Linear,
    rank_fraction: float,
    min_rank: int = 1,
) -> LowRankLinear:
    """
    Replace a Linear layer by a dense low-rank factorization initialized by SVD.

    This is weight-only post-hoc compression. It does not use activations yet.
    """
    if not (0.0 < rank_fraction <= 1.0):
        raise ValueError(f"rank_fraction must be in (0, 1], got {rank_fraction}")

    W = linear.weight.detach()
    out_features, in_features = W.shape

    max_rank = min(out_features, in_features)
    rank = max(min_rank, int(round(rank_fraction * max_rank)))
    rank = min(rank, max_rank)

    compressed = LowRankLinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=linear.bias is not None,
        device=W.device,
        dtype=W.dtype,
    )

    # Full SVD is fine for current LeWM dimensions.
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)

    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # W ≈ (U_r diag(S_r)) @ Vh_r
    A = U_r * S_r.unsqueeze(0)  # (out, rank)
    B = Vh_r                   # (rank, in)

    compressed.down.weight.data.copy_(B.to(dtype=W.dtype, device=W.device))
    compressed.up.weight.data.copy_(A.to(dtype=W.dtype, device=W.device))

    if linear.bias is not None:
        compressed.up.bias.data.copy_(linear.bias.detach())

    return compressed


def _get_child(parent: nn.Module, child_name: str) -> nn.Module:
    return getattr(parent, child_name)


def _set_child(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    setattr(parent, child_name, child)


def iter_named_linears(module: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            yield name, child


def replace_linears_with_svd(
    module: nn.Module,
    rank_fraction: float,
    min_rank: int = 1,
    skip_names: set[str] | None = None,
) -> tuple[nn.Module, LowRankReport]:
    """
    Deep-copy `module` and replace all selected nn.Linear layers by SVD low-rank layers.

    This is generic. It can be applied to LeWM predictor, pred_proj, etc.

    skip_names contains full module names relative to `module`, e.g.:
        {"some_block.linear"}
    """
    skip_names = skip_names or set()

    original_params = count_parameters(module)
    compressed = deepcopy(module)

    # Need list first because we mutate modules during replacement.
    linear_names = [
        name for name, child in compressed.named_modules()
        if isinstance(child, nn.Linear) and name not in skip_names
    ]

    num_replaced = 0

    for full_name in linear_names:
        parts = full_name.split(".")
        parent = compressed
        for part in parts[:-1]:
            parent = _get_child(parent, part)

        child_name = parts[-1]
        linear = _get_child(parent, child_name)

        if not isinstance(linear, nn.Linear):
            continue

        low_rank = svd_compress_linear(
            linear,
            rank_fraction=rank_fraction,
            min_rank=min_rank,
        )
        _set_child(parent, child_name, low_rank)
        num_replaced += 1

    compressed_params = count_parameters(compressed)

    report = LowRankReport(
        method="svd_low_rank_dense",
        target="generic_module",
        rank_fraction=float(rank_fraction),
        min_rank=int(min_rank),
        num_linear_replaced=int(num_replaced),
        original_target_params=int(original_params),
        compressed_target_params=int(compressed_params),
        target_param_ratio=float(compressed_params / original_params),
        target_param_reduction=float(1.0 - compressed_params / original_params),
    )

    return compressed, report