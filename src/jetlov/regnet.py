import torch
import torch.nn as nn


class RegNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        hidden: int = 128,
        output_dim: int = 5,
        dropout: float = 0.1,
    ):
        super(RegNet, self).__init__()
        self.network = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim),
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.network:
            x = layer(x)
            if isinstance(layer, nn.ReLU) and self.training:
                x = self.dropout(x)
        return x
