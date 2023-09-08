import torch.nn as nn
import torch.nn.functional as F
from jetron.gcnlayer import GCNLayer

class JetronNet(nn.Module):
    def __init__(self):
        super(JetronNet, self).__init__()
        self.layer1 = GCNLayer(4, 32)
        self.layer2 = GCNLayer(32, 32)
        self.layer3 = GCNLayer(32, 5)
        self.bn_fts = nn.BatchNorm1d(4) 

    def forward(self, g, features):
        feat = self.bn_fts(features)
        x = F.relu(self.layer1(g, feat))
        x = F.relu(self.layer2(g, x))
        x = self.layer3(g, x)
        return x
