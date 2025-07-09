from model import *


class Classifier(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_classes: int,
            normalize: Type[nn.Module] = nn.LayerNorm,
            activation: Type[nn.Module] = nn.Sigmoid,
            dropout: float = 0.0
    ) -> None:
        super(Classifier, self).__init__()
        # Sequential layer
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            normalize(hidden_size),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        # Initialize weights
        for p in self.layer.parameters():
            if isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight)
                if p.bias is not None:
                    nn.init.zeros_(p.bias)
            if isinstance(p, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(p.weight)
                nn.init.zeros_(p.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)

