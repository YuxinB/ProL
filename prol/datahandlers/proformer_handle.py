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
        self.contextlength = args.proformer['contextlength']
        self.t = len(y)
        self.time = torch.arange(self.t).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        r = np.random.randint(0, self.t-2*self.contextlength) # select the start of the history
        if self.args.proformer["multihop"]:
            s = np.random.randint(r+self.contextlength, r+2*self.contextlength)  # select a 'future' datum
        else:
            s = r+self.contextlength  # select the next datum

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
        self.contextlength = args.proformer["contextlength"]
        self.train_data = x_train[-self.contextlength:]
        self.train_targets = y_train[-self.contextlength:]
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

class VisionSequentialDataset(Dataset):
    def __init__(self, args, dataset, seqInd, maplab):
        """Create a dataset of context window (history + single future datum)

        Parameters
        ----------
        dataset : _type_
            original torch dataset
        seqInd : _type_
            training sequence indices
        maplab : _type_
            label mapper
        contextlength : int, optional
            length of the history
        """
        self.args = args
        self.dataset = dataset
        self.contextlength = args.proformer["contextlength"]
        self.t = len(seqInd)
        self.time = torch.arange(self.t).float()
        self.seqInd = seqInd
        self.maplab = maplab

    def __len__(self):
        return len(self.seqInd)

    def __getitem__(self, idx):
        r = np.random.randint(0, len(self.seqInd)-2*self.contextlength) # select the start of the history
        if self.args.proformer["multihop"]:
            s = np.random.randint(r+self.contextlength, r+2*self.contextlength)  # select a 'future' datum
        else:
            s = r+self.contextlength  # select the next datum

        id = list(range(r, r+self.contextlength)) + [s]
        dataid = self.seqInd[id] # get indices for the context window

        data = self.dataset.data[dataid]
        data = data.flatten(1, -1)
        labels = self.dataset.targets[dataid].apply_(self.maplab)
        time = self.time[id]
        
        target = labels[-1].clone() # true label of the future datum
        labels[-1] = np.random.binomial(1, 0.5) # replace the true label of the future datum with a random label

        return data, time, labels, target
    
class VisionSequentialTestDataset(Dataset):
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
        self.contextlength = args.proformer["contextlength"]
        self.train_seqInd = train_seqInd[-self.contextlength:]
        self.test_seqInd = test_seqInd[t:]
        self.maplab = maplab
        
        self.train_time = torch.arange(t).float()
        self.test_time = torch.arange(t, t + len(test_seqInd)).float()
        
    def __len__(self):
        return len(self.test_seqInd)
        
    def __getitem__(self, idx):
        dataid = self.train_seqInd.tolist() + [self.test_seqInd[idx]] # most recent history + inference datum indices

        data = self.dataset.data[dataid]
        data = data.flatten(1, -1)
        labels = self.dataset.targets[dataid].apply_(self.maplab)
        time = torch.cat([
            self.train_time[-self.contextlength:], 
            self.test_time[idx].view(1)
        ])

        target = labels[-1].clone() # true label of the future datum
        labels[-1] = np.random.binomial(1, 0.5) # replace the true label of the future datum with a random label

        return data, time, labels, target
