import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
from .base_trainer import BaseTrainer

class Model(nn.Module):
    """
    multi-layer perceptron
    """
    def __init__(self, num_classes=10, input_size=784, hidden_size=512):
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
def model_defaults(dataset):
    if dataset == 'mnist':
        return {
            "input_size": 28*28,
            "hidden_size": 512
        }
    elif dataset == 'cifar-10':
        return {
            "input_size": 32*32*3,
            "hidden_size": 512
        }
    elif dataset == 'synthetic':
        return {
            "input_size": 1,
            "hidden_size": 256
        }
    else:
        raise NotImplementedError

class SequentialDataset(Dataset):
    def __init__(self, args, dataset, seqInd, maplab):
        """Create the training dataset

        Parameters
        ----------
        dataset : _type_
            original torch dataset
        seqInd : _type_
            training sequence indices
        maplab : _type_
            label mapper
        """
        self.args = args
        self.dataset = dataset
        self.seqInd = seqInd
        self.maplab = maplab

    def __len__(self):
        return len(self.seqInd)

    def __getitem__(self, idx):
        data = self.dataset.data[self.seqInd[idx]].flatten(0, -1)
        label = self.dataset.targets[self.seqInd[idx]].apply_(self.maplab)
        return data, label

class SequentialTestDataset(Dataset):
    def __init__(self, args, dataset, train_seqInd, test_seqInd, maplab) -> None:
        """Create the testing dataset

        Parameters
        ----------
        args : _type_
            _description_
        dataset : _type_
            original torch dataset
        train_seqInd : _type_
            training sequence indices
        test_seqInd : _type_
            testing sequence indices
        maplab : _type_
            label mapper
        """
        t = len(train_seqInd)
        self.dataset = dataset
        self.test_seqInd = test_seqInd[t:]
        self.maplab = maplab
        
    def __len__(self):
        return len(self.test_seqInd)
        
    def __getitem__(self, idx):
        data = self.dataset.data[self.test_seqInd[idx]].flatten(0, -1)
        label = self.dataset.targets[self.test_seqInd[idx]].apply_(self.maplab)
        return data, label
    
class Trainer(BaseTrainer):
    def __init__(self, model, dataset, args) -> None:
        super().__init__(model, dataset, args)
        
    
def main():
    # testing
    x = torch.randn(1, 28*28)
    net = Model(2, 28*28, 512)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    main()