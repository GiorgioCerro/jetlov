import torch.nn as nn
from torch import Tensor
from dgl import DGLGraph
from typing import Union, Tuple

class Composite(nn.Module):
    def __init__(self, network_a: nn.Module, network_b: nn.Module, 
            return_lund: bool=False):
        super(Composite, self).__init__()
        self.network_a = network_a
        self.network_b = network_b
        self.return_lund = return_lund


    def forward(self, graph: DGLGraph) -> Union[Tensor, Tuple[Tensor]]:
        x_a = self.network_a(graph.ndata["coordinates"])
        graph.ndata["features"] = x_a
        x_b = self.network_b(graph)
        if self.return_lund:
            return x_b, x_a
        else:
            return x_b
