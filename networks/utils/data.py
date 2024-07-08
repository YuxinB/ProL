import numpy as np
import pickle
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SyntheticScenario2:
    """
    Create multiple sequences for Scenario 2
    """
    def __init__(self, cfg):
        self.seq_len = cfg.seq_len
        self.num_seeds = cfg.num_seeds
        self.period = cfg.period
        self.cfg = cfg

    def generate_data(self):
        xseq, yseq, taskseq = [], [], []
        tseq = []
        for sd in range(self.num_seeds):
            dat = self.gen_sequence(sd)
            xseq.append(dat[0])
            yseq.append(dat[1])
            taskseq.append(dat[2])
            tseq.append(np.arange(self.seq_len))

        xseq = np.array(xseq)
        yseq = np.array(yseq)
        tseq = np.array(tseq)
        taskseq = np.array(taskseq)

        self.data = {'x': xseq,
                     'y': yseq,
                     't': tseq,
                     'task': taskseq,
                     'cfg': self.cfg}

    def gen_sequence(self, seed):
        np.random.seed(seed)

        # generate a samples from from U[-2, -1] union U[2, 1]
        x1 = np.random.uniform(-2, -1, self.seq_len)
        x2 = np.random.uniform(1, 2, self.seq_len)
        mask = np.random.choice([0, 1], p=[0.5, 0.5], size=self.seq_len)
        Xdat = x1 * mask + x2 * (1 - mask)

        # Create labels
        T = self.period
        tind = np.array((np.arange(0, self.seq_len) % T) < (T // 2))
        tind = tind.astype(int)
        ind = np.where(tind == 1)[0]

        Ydat = Xdat > 0
        Ydat[ind] = Xdat[ind] < 0
        Ydat = Ydat.astype(int)
        
        Xdat = Xdat.reshape(-1, 1)

        return Xdat, Ydat, tind

    def store_data(self):
        os.makedirs('data/synthetic', exist_ok=True)
        with open('data/synthetic/scenario2_period%d.pkl' % self.period, 'wb') as fp:
            pickle.dump(self.data, fp)


class SyntheticDataset(Dataset):
    def __init__(self, data_path, idx, run_id, test, past=None):
        with open(data_path, 'rb') as fp:
            self.data = pickle.load(fp)
        self.x = torch.FloatTensor(self.data['x']) // 4
        self.y = torch.LongTensor(self.data['y'])
        self.t = torch.FloatTensor(self.data['t'])

        if test:
            self.x = self.x[run_id, :]
            self.y = self.y[run_id, :]
            self.t = self.t[run_id, :]
        else:
            if past is None:
                # Create dataloader with full history
                self.x = self.x[run_id, :idx]
                self.y = self.y[run_id, :idx]
                self.t = self.t[run_id, :idx]
            else:
                # Create dataloader with just recent history
                self.x = self.x[run_id, idx-past:idx]
                self.y = self.y[run_id, idx-past:idx]
                self.t = self.t[run_id, idx-past:idx]
       
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        t = self.t[idx]
        return x, y, t


def create_dataloader(cfg, t, seed):
    past = cfg.fine_tune
    train_dataset = SyntheticDataset(cfg.data.path, t, seed, False, past)
    test_dataset = SyntheticDataset(cfg.data.path, t, seed, True, past)

    trainloader = DataLoader(train_dataset,
                            batch_size=cfg.data.bs,
                            shuffle=True, pin_memory=True,
                            num_workers=cfg.data.workers)
    testloader = DataLoader(test_dataset, batch_size=500,
                            shuffle=False, pin_memory=True,
                            num_workers=cfg.data.workers)

    return trainloader, testloader
