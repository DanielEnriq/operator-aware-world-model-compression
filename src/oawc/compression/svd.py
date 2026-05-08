from __future__ import annotations

import torch
from torch import nn

from oawc.compression.factorized import FactorizedLinear


def factorized_linear_param_count(
    in_features: int,
    out_features: int,
    rank: int,
    *,
    with_bias: bool,
) -> int:
    bias_params = out_features if with_bias else 0
    return int(rank * (in_features + out_features) + bias_params)


def relative_fro_error(
    original_weight: torch.Tensor,
    approx_weight: torch.Tensor,
) -> float:
    original = original_weight.float()
    approx = approx_weight.float()
    denom = torch.linalg.norm(original, ord="fro")
    if float(denom) == 0.0:
        return 0.0
    num = torch.linalg.norm(original - approx, ord="fro")
    return float((num / denom).item())


def factorize_linear_svd(layer: nn.Linear, rank: int) -> FactorizedLinear:
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)!r}")

    out_features, in_features = layer.weight.shape
    max_rank = min(in_features, out_features)
    if rank < 1 or rank >= max_rank:
        raise ValueError(
            f"Invalid rank={rank} for linear({in_features},{out_features}). "
            f"Expected 1 <= rank < {max_rank}."
        )

    ref_param = layer.weight
    factorized = FactorizedLinear(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        bias=layer.bias is not None,
        device=ref_param.device,
        dtype=ref_param.dtype,
    )

    # SVD in float32 for numerical stability.
    w_float = layer.weight.detach().float()
    u, s, vh = torch.linalg.svd(w_float, full_matrices=False)

    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]

    sqrt_s = torch.sqrt(torch.clamp(s_r, min=0.0))
    down_w = torch.diag(sqrt_s) @ vh_r
    up_w = u_r @ torch.diag(sqrt_s)

    factorized.down.weight.data.copy_(
        down_w.to(device=ref_param.device, dtype=ref_param.dtype)
    )
    factorized.up.weight.data.copy_(
        up_w.to(device=ref_param.device, dtype=ref_param.dtype)
    )
    if layer.bias is not None:
        factorized.up.bias.data.copy_(
            layer.bias.detach().to(device=ref_param.device, dtype=ref_param.dtype)
        )

    return factorized
