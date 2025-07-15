import torch.nn as nn
from ncps.torch import LTC, CfC
from model.classifier import Classifier
from typing import List
from torch import Tensor
from model.utils import forward


class MultiHeadLTC(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,
            num_classes: int
    ) -> None:
        super(MultiHeadLTC, self).__init__()
        # Liquid time-constant networks (LTC)
        self.cell = nn.ModuleList([
            LTC(
                input_size=1,
                units=window_size,
                batch_first=True
            )
            for _ in range(num_variables)
        ])

        # Classifier
        self.fc = Classifier(
            input_size=window_size * num_variables,
            hidden_size=hidden_size,
            num_classes=num_classes
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        return forward(x, self.cell, self.fc)


class MultiHeadCfC(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,
            num_classes: int
    ) -> None:
        super(MultiHeadCfC, self).__init__()
        # Closed-form continuous-time neural networks (CfC)
        self.cell = nn.ModuleList([
            CfC(
                input_size=1,
                units=window_size,
                batch_first=True
            )
            for _ in range(num_variables)
        ])

        # Classifier
        self.fc = Classifier(
            input_size=window_size * num_variables,
            hidden_size=hidden_size,
            num_classes=num_classes
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        return forward(x, self.cell, self.fc)

