from model import *


def cell(x: List[Tensor], cell: nn.ModuleList, fc: nn.Sequential) -> Tensor:
    outputs: List[Tensor] = []

    for c, v in zip(cell, x):
        out, _ = c(v)
        outputs.append(out[:, -1, :])

    concat = torch.cat(outputs, dim=1)
    return fc(concat)

