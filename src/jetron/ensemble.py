import torch.nn as nn

class Ensemble(nn.Module):
    def __init__(self, network_a, network_b, return_lund=False):
        super(Ensemble, self).__init__()
        self.network_a = network_a
        self.network_b = network_b
        self.return_lund = return_lund


    def forward(self, graph):
        x_a = self.network_a(graph.ndata["coordinates"])
        graph.ndata["features"] = x_a
        x_b = self.network_b(graph)
        if self.return_lund:
            return x_b, x_a
        else:
            return x_b
