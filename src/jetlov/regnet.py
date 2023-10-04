import torch
import torch.nn as nn

class RegNet(nn.Module):
    def __init__(self, input_dim=8, hidden=128, output_dim=5):
        super(RegNet, self).__init__()
        self.branches = nn.ModuleList()
        for idx in range(output_dim):
            if idx == 0 or idx == 3:
                # smaller branch for lnK and lnMass
                self.branches.append(nn.Sequential(
                                nn.Linear(input_dim, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1)))
            else:
                self.branches.append(nn.Sequential(
                                nn.Linear(input_dim, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1)))

    
    def forward(self, x):
        outputs = []
        for idx, branch in enumerate(self.branches):
            outputs.append(branch(x))
        return torch.concat(outputs, axis=1)
