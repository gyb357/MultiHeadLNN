import torch.nn as nn
import torch
from model.utils import get_rnn_layer, init_weights
from model.module import Classifier
from typing import List
from torch import Tensor


class MultiHead(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,  # MLP Classifier hidden_size
            num_classes: int,
            rnn_type: str
    ) -> None:
        super(MultiHead, self).__init__()
        # RNN layers
        self.rnn = nn.ModuleList([
            get_rnn_layer(
                rnn_type=rnn_type,
                input_size=1,
                hidden_size=window_size  # RNN hidden_size
            )
            for _ in range(num_variables)
        ])

        # MLP Classifier
        self.fc = Classifier(
            input_size=window_size * num_variables,
            hidden_size=hidden_size,
            num_classes=num_classes
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: List[Tensor]) -> Tensor:
        outs: List[Tensor] = []

        for rnn, var in zip(self.rnn, x):
            seq, _ = rnn(var)
            outs.append(seq[:, -1, :])   # (B, W) - last time step
        concat = torch.cat(outs, dim=1)  # (B, W * V)
        return self.fc(concat)


class SingleHead(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,  # MLP Classifier hidden_size
            num_classes: int,
            rnn_type: str
    ) -> None:
        super(SingleHead, self).__init__()
        # RNN layers
        self.rnn = get_rnn_layer(
            rnn_type=rnn_type,
            input_size=num_variables,
            hidden_size=window_size  # RNN hidden_size
        )

        # MLP Classifier
        self.fc = Classifier(
            input_size=window_size,
            hidden_size=hidden_size,
            num_classes=num_classes
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: List[Tensor]) -> Tensor:
        x = torch.cat(x, dim=2)        # (B, W, 1) * V
        seq, _ = self.rnn(x)
        return self.fc(seq[:, -1, :])  # (B, W) - last time step


class ANN(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,     # MLP Classifier hidden_size
            num_classes: int,
            rnn_type: str = None  # Not used, for compatibility
    ) -> None:
        super(ANN, self).__init__()
        # MLP Classifier
        self.fc = Classifier(
            input_size=window_size * num_variables,
            hidden_size=hidden_size,
            num_classes=num_classes
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: List[Tensor]) -> Tensor:
        x = torch.cat(x, dim=2)  # (B, W, 1) * V
        B, W, V = x.shape
        return self.fc(x.reshape(B, W * V))

