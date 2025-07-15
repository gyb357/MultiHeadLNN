import torch.nn as nn
import torch
from typing import List
from torch import Tensor


def forward(x: List[Tensor], cell: nn.ModuleList, fc: nn.Sequential) -> Tensor:
    outputs: List[Tensor] = []

    for c, v in zip(cell, x):
        out, _ = c(v)
        outputs.append(out[:, -1, :])

    concat = torch.cat(outputs, dim=1)
    return fc(concat)


def get_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

