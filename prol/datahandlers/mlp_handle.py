import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticSequentialDataset(Dataset):
    def __init__(self, args, x, y):
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
        self.data = x
        self.targets = y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx].unsqueeze(-1)
        label = self.targets[idx]
        return data, label

class SyntheticSequentialTestDataset(Dataset):
    def __init__(self, args, x_train, y_train, x, y) -> None:
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
        t = len(y_train)
        self.test_data = x[t:]
        self.test_targets = y[t:]
        
    def __len__(self):
        return len(self.test_targets)
        
    def __getitem__(self, idx):
        data = self.test_data[idx].unsqueeze(-1)
        label = self.test_targets[idx]
        return data, label
