import torch
from torch.utils.data import Dataset

class VisionSequentialDataset(Dataset):
    def __init__(self, args, dataset, transform, seqInd, maplab):
        """Create a dataset of context window (history + single future datum)

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
        self.t = len(seqInd)
        self.time = torch.arange(self.t).float()
        self.maplab = maplab
        self.transform = transform

    def __len__(self):
        return len(self.seqInd)

    def __getitem__(self, idx):
        data = self.dataset.data[self.seqInd[idx]]
        data = self.transform(data)
        label = self.dataset.targets[self.seqInd[idx]].apply_(self.maplab)
        time = self.time[idx]
        return data, time, label

class VisionSequentialTestDataset(Dataset):
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
    def __init__(self, args, dataset, transform, train_seqInd, test_seqInd, maplab) -> None:
        t = len(train_seqInd)
        self.dataset = dataset
        self.test_seqInd = test_seqInd[t:]
        self.maplab = maplab
        self.transform = transform

        self.test_time = torch.arange(t, t + len(test_seqInd)).float()
        
    def __len__(self):
        return len(self.test_seqInd)
        
    def __getitem__(self, idx):
        data = self.dataset.data[self.test_seqInd[idx]]
        data = self.transform(data)
        label = self.dataset.targets[self.test_seqInd[idx]].apply_(self.maplab)
        time = self.test_time[idx]
        return data, time, label
