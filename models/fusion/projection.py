import torch.nn as nn

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=512):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.fc(x)))
