from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def params_matching_substring(
    module: nn.Module,
    substring: str,
) -> int:
    return int(
        sum(
            p.numel()
            for name, p in module.named_parameters()
            if substring in name
        )
    )


def model_size_bytes(module: nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return int(total)


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2))
