from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from oawc.compression.factorized import FactorizedLinear
from oawc.compression.svd import (
    factorize_linear_svd,
    factorized_linear_param_count,
    relative_fro_error,
)


@dataclass
class ActivationAwareResult:
    factorized: FactorizedLinear
    w_tilde: torch.Tensor
    relative_weight_error: float
    relative_activation_output_error: float
    ridge_used: float
    cholesky_retries: int
    used_fallback_weight_svd: bool


def _relative_activation_error(
    x_rows: torch.Tensor,
    w: torch.Tensor,
    w_tilde: torch.Tensor,
) -> float:
    x = x_rows.float()
    y = x @ w.float().T
    y_tilde = x @ w_tilde.float().T
    denom = torch.linalg.norm(y, ord="fro")
    if float(denom.item()) == 0.0:
        return 0.0
    num = torch.linalg.norm(y - y_tilde, ord="fro")
    return float((num / denom).item())


def _compute_scaled_ridge(
    xtx_over_n: torch.Tensor,
    ridge: float,
) -> float:
    d = xtx_over_n.shape[0]
    trace_val = torch.trace(xtx_over_n)
    scaled = float(ridge)
    if d > 0 and torch.isfinite(trace_val):
        trace_f = float(trace_val.item())
        if trace_f > 0.0:
            scaled = float(ridge * (trace_f / d))
    return scaled


def _activation_aware_weight_tilde(
    weight: torch.Tensor,
    x_rows: torch.Tensor,
    rank: int,
    ridge: float,
    max_retries: int = 3,
) -> tuple[torch.Tensor, float, int]:
    # Work entirely in float32 for numerical stability.
    w = weight.float()
    x = x_rows.float()
    n = max(1, x.shape[0])
    d_in = x.shape[1]

    xtx_over_n = (x.T @ x) / float(n)
    eye = torch.eye(d_in, device=x.device, dtype=x.dtype)

    base_ridge = _compute_scaled_ridge(xtx_over_n, ridge)
    ridge_used = base_ridge
    retries = 0
    chol = None

    for attempt in range(max_retries + 1):
        ridge_try = base_ridge * (10.0 ** attempt)
        g = xtx_over_n + ridge_try * eye
        try:
            chol = torch.linalg.cholesky(g)
            ridge_used = ridge_try
            retries = attempt
            break
        except RuntimeError:
            continue

    if chol is None:
        raise RuntimeError("Cholesky failed after ridge retries.")

    # M = W C
    m = w @ chol
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]
    m_r = (u_r * s_r.unsqueeze(0)) @ vh_r

    # Solve W_tilde @ C = M_r  -> W_tilde = solve(C^T, M_r^T)^T
    w_tilde = torch.linalg.solve(chol.T, m_r.T).T
    return w_tilde, float(ridge_used), int(retries)


def factorize_linear_activation_aware(
    layer: nn.Linear,
    x_rows: torch.Tensor,
    rank: int,
    ridge: float,
) -> ActivationAwareResult:
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer)!r}")
    if x_rows.ndim != 2 or x_rows.shape[1] != layer.in_features:
        raise ValueError(
            "x_rows must have shape [N, in_features], got "
            f"{tuple(x_rows.shape)} for in_features={layer.in_features}"
        )

    w = layer.weight.detach()
    w_tilde, ridge_used, retries = _activation_aware_weight_tilde(
        weight=w,
        x_rows=x_rows,
        rank=rank,
        ridge=ridge,
    )

    temp = nn.Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        device=w.device,
        dtype=w.dtype,
    )
    temp.weight.data.copy_(w_tilde.to(device=w.device, dtype=w.dtype))
    if layer.bias is not None:
        temp.bias.data.copy_(layer.bias.detach())
    factorized = factorize_linear_svd(temp, rank=rank)

    w_approx = (
        factorized.up.weight.detach().float()
        @ factorized.down.weight.detach().float()
    )
    rel_w = relative_fro_error(w.float(), w_approx)
    rel_act = _relative_activation_error(x_rows, w, w_approx)

    return ActivationAwareResult(
        factorized=factorized,
        w_tilde=w_tilde.detach().cpu(),
        relative_weight_error=rel_w,
        relative_activation_output_error=rel_act,
        ridge_used=ridge_used,
        cholesky_retries=retries,
        used_fallback_weight_svd=False,
    )


def factorize_linear_activation_aware_with_fallback(
    layer: nn.Linear,
    x_rows: torch.Tensor,
    rank: int,
    ridge: float,
) -> ActivationAwareResult:
    try:
        return factorize_linear_activation_aware(
            layer=layer,
            x_rows=x_rows,
            rank=rank,
            ridge=ridge,
        )
    except RuntimeError:
        factorized = factorize_linear_svd(layer, rank=rank)
        w = layer.weight.detach().float()
        w_approx = (
            factorized.up.weight.detach().float()
            @ factorized.down.weight.detach().float()
        )
        rel_w = relative_fro_error(w, w_approx)
        rel_act = _relative_activation_error(x_rows, w, w_approx)
        return ActivationAwareResult(
            factorized=factorized,
            w_tilde=w_approx.detach().cpu(),
            relative_weight_error=rel_w,
            relative_activation_output_error=rel_act,
            ridge_used=float(ridge),
            cholesky_retries=4,
            used_fallback_weight_svd=True,
        )


def candidate_layer_rank(
    in_features: int,
    out_features: int,
    rank_fraction: float,
) -> int:
    rank = int(rank_fraction * min(in_features, out_features))
    rank = max(1, rank)
    if rank >= min(in_features, out_features):
        rank = min(in_features, out_features) - 1
    return int(rank)


def is_compressive_linear(
    in_features: int,
    out_features: int,
    rank: int,
    has_bias: bool,
) -> bool:
    before = int(
        in_features * out_features + (out_features if has_bias else 0)
    )
    after = factorized_linear_param_count(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        with_bias=has_bias,
    )
    return bool(after < before)


def append_activation_rows(
    existing_rows: torch.Tensor | None,
    new_rows: torch.Tensor,
    *,
    max_rows: int,
    generator: torch.Generator,
) -> torch.Tensor:
    rows = new_rows.detach().float().cpu()
    if rows.ndim != 2:
        raise ValueError(
            f"Expected rank-2 rows, got shape={tuple(rows.shape)}"
        )

    if existing_rows is None:
        if rows.shape[0] <= max_rows:
            return rows.clone()
        idx = torch.randperm(rows.shape[0], generator=generator)[:max_rows]
        return rows[idx]

    combined = torch.cat([existing_rows, rows], dim=0)
    if combined.shape[0] <= max_rows:
        return combined
    idx = torch.randperm(combined.shape[0], generator=generator)[:max_rows]
    return combined[idx]


def linear_param_count(layer: nn.Linear) -> int:
    return int(
        layer.weight.numel()
        + (layer.bias.numel() if layer.bias is not None else 0)
    )


def module_param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))
