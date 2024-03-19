import torch
import torch.nn as nn
from jetlov.hyp_layers import HNNLayer
from jetlov.manifolds.euclidean import Euclidean

class HRegNet(nn.Module):
    def __init__(self, input_dim: int=8, hidden: int=128, output_dim: int=5, 
            dropout: float=0.):
        super(HRegNet, self).__init__()
        manifold = Euclidean()
        self.network = nn.Sequential(
                HNNLayer(manifold=manifold, in_features=input_dim, 
                    out_features=hidden, c=1, dropout=dropout, act=nn.ReLU(),
                    use_bias=True), 
                HNNLayer(manifold=manifold, in_features=hidden, 
                    out_features=output_dim, c=1, dropout=dropout, act=nn.ReLU(),
                    use_bias=True), 
        )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
