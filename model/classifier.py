import torch.nn as nn
from typing import Type
from torch import Tensor


class Classifier(nn.Module):
    normalize: Type[nn.Module] = nn.LayerNorm # nn.BatchNorm1d,
    activation: Type[nn.Module] = nn.Sigmoid  # nn.Tanh, nn.ReLU, nn.GeLU, ...

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            bias: bool = True,
            dropout: float = 0.0
    ) -> None:
        super(Classifier, self).__init__()
        # Sequential layer
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias),
            self.normalize(hidden_size),
            self.activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes, bias)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

