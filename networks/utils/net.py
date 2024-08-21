import torch
import numpy as np
import torch.nn as nn


def create_net(cfg):
    if cfg.net.type == 'mlp':
        net = MLP(1, 2, 256)
    elif cfg.net.type == 'mlp3':
        net = MLP(2, 2, 256)
    elif cfg.net.type == 'mlp_mnist':
        net = MLP(784, 5, 256)
    elif cfg.net.type == 'prospective_mlp':
        net = ProspectiveMLP(cfg, 1, 2, 256)
    elif cfg.net.type == 'prospective_mlp3':
        net = ProspectiveMLP(cfg, 2, 2, 256)
    elif cfg.net.type == 'prospective_mlp_mnist':
        net = ProspectiveMLP(cfg, 784, 5, 256)
    elif cfg.net.type == 'cnn_cifar':
        net = CNN(5)
    elif cfg.net.type == 'prospective_cnn_cifar':
        net = ProspectiveCNN(cfg, 5)
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


class CNN(nn.Module):
    """
    Small convolution network with no residual connections (single-head)
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        channels = 3
        avg_pool = 2
        linsize = 320
        self.conv1 = nn.Conv2d(channels, 80, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(avg_pool)

        self.fc = nn.Linear(linsize, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = self.conv2(x)
        x = self.maxpool(self.relu(self.bn2(x)))

        x = self.conv3(x)
        x = self.maxpool(self.relu(self.bn3(x)))
        x = x.flatten(1, -1)

        x = self.fc(x)
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


class ProspectiveCNN(nn.Module):
    """
    Small convolution network with no residual connections (single-head)
    """
    def __init__(self, cfg, num_classes=10, tdim=50):
        super(ProspectiveCNN, self).__init__()
        channels = 3
        avg_pool = 2
        linsize = 320
        self.time_embed = TimeEmbedding(cfg.dev, tdim)
        self.time_project = nn.Linear(tdim, 32*32)

        channels += 1

        self.conv1 = nn.Conv2d(channels, 80, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(avg_pool)
        self.fc = nn.Linear(linsize, num_classes)

    def forward(self, x, t):
        tembed = self.time_embed(t.reshape(-1, 1))
        tembed = self.time_project(tembed).view(-1, 1, 32, 32)

        x = torch.cat([x, tembed], dim=1)

        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = self.conv2(x)
        x = self.maxpool(self.relu(self.bn2(x)))

        x = self.conv3(x)
        x = self.maxpool(self.relu(self.bn3(x)))
        x = x.flatten(1, -1)

        x = self.fc(x)
        return x
