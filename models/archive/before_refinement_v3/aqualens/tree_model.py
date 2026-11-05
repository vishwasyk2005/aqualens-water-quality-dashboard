"""DET: temporal GRU and genuine Child-Sum Tree-LSTM composition."""
import torch
from torch import nn

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.iou = nn.Linear(hidden, hidden * 3)
        self.forget = nn.Linear(hidden, hidden)
    def forward(self, h, c):
        i, o, u = self.iou(h.sum(1)).chunk(3, -1)
        memory = torch.sigmoid(i) * torch.tanh(u) + (torch.sigmoid(self.forget(h)) * c).sum(1)
        return torch.sigmoid(o) * torch.tanh(memory), memory

class HierarchicalForecaster(nn.Module):
    """A single DET; three direct forecast heads at 1/2/4 native observations."""
    def __init__(self, cell_zone, hidden=24, features=12):
        super().__init__()
        self.cell_zone = list(cell_zone)
        self.zone_count = max(cell_zone) + 1
        self.encoder = nn.GRU(features, hidden, batch_first=True)
        self.leaf_memory = nn.Linear(hidden, hidden)
        self.zone_tree = ChildSumTreeLSTM(hidden)
        self.lake_tree = ChildSumTreeLSTM(hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 9))
        self.zone_indices = [[j for j, p in enumerate(cell_zone) if p == z] for z in range(self.zone_count)]
    def forward(self, x):
        batch, steps, cells, features = x.shape
        _, encoded = self.encoder(x.permute(0, 2, 1, 3).reshape(batch * cells, steps, features))
        leaves = encoded[-1].reshape(batch, cells, -1)
        memories = torch.tanh(self.leaf_memory(leaves))
        states = [self.zone_tree(leaves[:, ix], memories[:, ix]) for ix in self.zone_indices]
        zh = torch.stack([v[0] for v in states], 1)
        zc = torch.stack([v[1] for v in states], 1)
        lake, _ = self.lake_tree(zh, zc)
        joined = torch.cat([zh, lake[:, None].expand(-1, self.zone_count, -1)], -1)
        # Residual prediction anchored in the most recent causal cell values.
        anchors = torch.stack([x[:, -1, ix, :3].mean(1) for ix in self.zone_indices], 1)
        residual = self.head(joined).reshape(batch, self.zone_count, 3, 3).permute(0, 2, 1, 3)
        return residual + anchors[:, None]

