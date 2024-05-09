import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticSequentialDataset(Dataset):
    def __init__(self, args, x, y):
        """Create a dataset of context window (history + single future datum)

        Parameters
        ----------
        args : _type_
            _description_
        x : _type_
            data
        y : _type_
            targets
        """
        self.args = args
        self.data = x
        self.targets = y
        self.contextlength = args.contextlength
        self.t = len(y)
        self.time = torch.arange(self.t).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        r = np.random.randint(0, self.t-2*self.contextlength) # select the start of the history
        if self.args.multihop:
            s = np.random.randint(r+self.contextlength, r+2*self.contextlength)  # select a 'future' datum
        else:
            s = r+self.contextlength  # select a 'future' datum

        id = list(range(r, r+self.contextlength)) + [s]

        data = self.data[id].unsqueeze(-1)
        labels = self.targets[id]
        time = self.time[id]
        
        target = labels[-1].clone() # true label of the future datum
        labels[-1] = np.random.binomial(1, 0.5) # replace the true label of the future datum with a random label

        return data, time, labels, target
    
class SyntheticSequentialTestDataset(Dataset):
    def __init__(self, args, x_train, y_train, x, y) -> None:
        """Create test dataset

        Parameters
        ----------
        args : _type_
            _description_
        x_train : _type_
            train data
        y_train : _type_
            train targets
        x : _type_
            test data
        y : _type_
            test targets
        """
        t = len(y_train)
        self.contextlength = args.contextlength
        self.train_data = x_train[-args.contextlength:]
        self.train_targets = y_train[-args.contextlength:]
        self.test_data = x[t:]
        self.test_targets = y[t:]
        
        self.train_time = torch.arange(t).float()
        self.test_time = torch.arange(t, t + len(y)).float()
        
    def __len__(self):
        return len(self.test_targets)
        
    def __getitem__(self, idx):
        # most recent history + inference datum indices
        data = torch.cat([
            self.train_data,
            self.test_data[idx].view(1)
        ]).unsqueeze(-1)
        labels = torch.cat([
            self.train_targets,
            self.test_targets[idx].view(1)
        ])
        time = torch.cat([
            self.train_time[-self.contextlength:], 
            self.test_time[idx].view(1)
        ])

        target = labels[-1].clone() # true label of the future datum
        labels[-1] = np.random.binomial(1, 0.5) # replace the true label of the future datum with a random label

        return data, time, labels, target
