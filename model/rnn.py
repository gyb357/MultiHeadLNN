import torch.nn as nn
from model.classifier import Classifier
from typing import List
from torch import Tensor
from model.utils import forward


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


class MultiHeadRNN(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,
            num_classes: int
    ) -> None:
        super(MultiHeadRNN, self).__init__()
        # Attributes
        self.hidden_size = hidden_size

        # RNN
        self.cell = nn.ModuleList([
            nn.RNN(input_size=1,
                    hidden_size=window_size,
                    batch_first=True
            )
            for _ in range(num_variables)
        ])

        # Classifier
        self.fc = Classifier(input_size=window_size * num_variables,
                             hidden_size=hidden_size,
                             num_classes=num_classes)

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: List[Tensor]) -> Tensor:
        return forward(x, self.cell, self.fc)


class MultiHeadLSTM(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,
            num_classes: int
    ) -> None:
        super(MultiHeadLSTM, self).__init__()
        # Attributes
        self.hidden_size = hidden_size

        # LSTM
        self.cell = nn.ModuleList([
            nn.LSTM(input_size=1,
                    hidden_size=window_size,
                    batch_first=True
            )
            for _ in range(num_variables)
        ])

        # Classifier
        self.fc = Classifier(input_size=window_size * num_variables,
                             hidden_size=hidden_size,
                             num_classes=num_classes)

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: List[Tensor]) -> Tensor:
        return forward(x, self.cell, self.fc)


class MultiHeadGRU(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,
            num_classes: int
    ) -> None:
        super(MultiHeadGRU, self).__init__()
        # Attributes
        self.hidden_size = hidden_size

        # GRU
        self.cell = nn.ModuleList([
            nn.GRU(input_size=1,
                   hidden_size=window_size,
                   batch_first=True
            )
            for _ in range(num_variables)
        ])

        # Classifier
        self.fc = Classifier(input_size=window_size * num_variables,
                             hidden_size=hidden_size,
                             num_classes=num_classes)

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: List[Tensor]) -> Tensor:
        return forward(x, self.cell, self.fc)

