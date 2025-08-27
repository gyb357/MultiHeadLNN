import torch.nn as nn
import torch
from model.utils import get_cell, init_weights
from model.module import Classifier
from typing import List
from torch import Tensor


class MultiHead(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,
            num_classes: int,
            cell_type: str
    ) -> None:
        super(MultiHead, self).__init__()
        # RNN Cell
        self.cell = nn.ModuleList([
            get_cell(type=cell_type,
                     input_size=1,
                     window_size=window_size,
                     batch_first=True)
            for _ in range(num_variables)
        ])

        # MLP Classifier
        self.fc = Classifier(input_size=window_size * num_variables,
                             hidden_size=hidden_size,
                             num_classes=num_classes)
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: List[Tensor]) -> Tensor:
        outputs: List[Tensor] = []

        for cell, var in zip(self.cell, x):
            out, _ = cell(var)
            outputs.append(out[:, -1, :])  # Last time step

        concat = torch.cat(outputs, dim=1) # (B, W * V)
        return self.fc(concat)


class SingleHead(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,
            num_classes: int,
            cell_type: str
    ) -> None:
        super(SingleHead, self).__init__()
        # RNN Cell
        self.cell = get_cell(type=cell_type,
                             input_size=num_variables,
                             window_size=window_size,
                             batch_first=True)
        
        # MLP Classifier
        self.fc = Classifier(input_size=window_size,
                             hidden_size=hidden_size,
                             num_classes=num_classes)
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: List[Tensor]) -> Tensor:
        x = torch.cat(x, dim=2)        # (B, W, 1) -> (B, W, V)
        out, _ = self.cell(x)
        return self.fc(out[:, -1, :])  # Last time step


class ANN(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,
            num_classes: int,
    ) -> None:
        super(ANN, self).__init__()
        # MLP Classifier
        self.fc = Classifier(input_size=window_size * num_variables,
                             hidden_size=hidden_size,
                             num_classes=num_classes)
        
        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        flatten: List[Tensor] = []

        for var in x:
            flatten.append(var.flatten(start_dim=1))

        concat = torch.cat(flatten, dim=1)
        return self.fc(concat)

