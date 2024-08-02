import torch
import numpy as np
import torch.nn as nn


def create_net(cfg):
    if cfg.net.type == 'mlp':
        net = MLP(1, 2, 256)
    elif cfg.net.type == 'mlp3':
        net = MLP(2, 2, 256)
    elif cfg.net.type == 'prospective_mlp':
        net = ProspectiveMLP(cfg, 1, 2, 256)
    elif cfg.net.type == 'prospective_mlp3':
        net = ProspectiveMLP(cfg, 2, 2, 256)
    else:
        raise NotImplementedError

    return net


class TimeEmbedding(nn.Module):
    def __init__(self, dev, dim):
        super(TimeEmbedding, self).__init__()
        self.freqs = (2 * np.pi) / (torch.arange(2, dim + 1, 2))
        self.freqs = self.freqs.unsqueeze(0).to(dev)

    def forward(self, t):
        self.sin = torch.sin(self.freqs * t)
        self.cos = torch.cos(self.freqs * t)

        return torch.cat([self.sin, self.cos], dim=-1)

class DiscreteTime(nn.Module):
    def __init__(self, dev, dim):
        super(DiscreteTime, self).__init__()
        self.mode = torch.arange(1, dim + 1).to(dev)

    def forward(self, t):
        t = (t % self.mode)
        t = torch.gt(t, self.mode // 2).float()
        return t

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, t):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ProspectiveMLP(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, hidden_dim, tdim=50):
        super(ProspectiveMLP, self).__init__()
        self.time_embed = TimeEmbedding(cfg.dev, tdim)
        # self.time_embed = DiscreteTime(cfg.dev, tdim)
        in_dim += tdim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, t):
        tembed = self.time_embed(t.reshape(-1, 1))
        x = torch.cat([x, tembed], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
