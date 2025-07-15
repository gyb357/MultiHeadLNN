import torch.nn as nn
import torch
from model.classifier import Classifier
from typing import List
from torch import Tensor


class ANN(nn.Module):
    def __init__(
            self,
            window_size: int,
            num_variables: int,
            hidden_size: int,
            num_classes: int
    ) -> None:
        super(ANN, self).__init__()
        # Classifier
        self.fc = Classifier(
            input_size=window_size * num_variables,
            hidden_size=hidden_size,
            num_classes=num_classes
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        flatten: list = []

        for v in x:
            v = v.squeeze(-1)
            flatten.append(v.flatten(start_dim=1))

        concat = torch.cat(flatten, dim=1)
        return self.fc(concat)

