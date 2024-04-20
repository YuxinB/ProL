import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import numpy as np

from prol.models.protransformer import TransformerClassifier

class SetParams:
    def __init__(self, dict) -> None:
        for k, v in dict.items():
            setattr(self, k, v)

def get_sub_dataset(dataset, classes=(0, 1)):
    subset = deepcopy(dataset)
    targets = subset.targets
    data = torch.cat((
        subset.data[targets == classes[0]], 
        subset.data[targets == classes[1]]
    ), axis=0)
    targets = torch.cat((
        targets[targets == classes[0]],
        targets[targets==classes[1]]
    )) 
    targets[targets == classes[0]] = 0
    targets[targets == classes[1]] = 1
    subset.data = data
    subset.targets = targets
    return subset

def get_data(dist, N, seed):
    np.random.seed(seed)
    idx = np.random.choice(len(dist), N, replace=False)
    return dist.data[idx], dist.targets[idx]

def get_data_sequence(N, t, dist_A, dist_B, seed=1996):
    seqData = []
    seqLab = []
    torch.manual_seed(1996)
    for i in range(t//N):
        if i % 2 == 0:
            data, targets = get_data(dist_A, N, seed + i * 200)
        else:
            data, targets = get_data(dist_B, N, seed + i * 1000)
        seqData.append(data)
        seqLab.append(targets)
    seqData = torch.cat(seqData, axis=0).view(t, data.shape[-1]**2).float()
    seqLab = torch.cat(seqLab, axis=0).long()
    return seqData, seqLab

class SequentialDataset(Dataset):
    def __init__(self, N, t, contextlength=200, seed=1996, train=True):
        dataset = torchvision.datasets.MNIST(
            root="../data",
            train=train,
            download=True
        )
        dist_A = get_sub_dataset(dataset, (0, 1))
        dist_B = get_sub_dataset(dataset, (2, 3))
        self.data, self.labels = get_data_sequence(N, t, dist_A, dist_B, seed=seed)
        self.contextlength = contextlength
        self.t = t
        self.time = torch.arange(t).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        r = np.random.randint(0, len(self.data)-2*self.contextlength) # select the end of the subsequence
        s = np.random.randint(r+self.contextlength, r+2*self.contextlength)  # select a 'future' time beyond the subsequence
        
        data = torch.cat((self.data[r:r+self.contextlength], self.data[s:s+1]), axis=0)
        labels = torch.cat((self.labels[r:r+self.contextlength], self.labels[s:s+1]), axis=0)
        time = torch.cat((self.time[r:r+self.contextlength], self.time[s:s+1]), axis=0)

        target = labels[-1].clone()
        labels[-1] = np.random.binomial(1, 0.5)

        return data, time, labels, target
    
class SequentialTestDataset(Dataset):
    def __init__(self, ) -> None:
        pass
        

class Trainer:
    def __init__(self, args) -> None:
        self.args = args

        dataset = SequentialDataset(
            N=args.N,
            t=args.t,
            contextlength=args.contextlength,
            seed=args.seed,
            train=True
        )
        self.trainloader = DataLoader(dataset, batch_size=32)

        self.model = TransformerClassifier(
            input_size=args.image_size ** 2,
            d_model=256, 
            num_heads=4,
            ff_hidden_dim=1024,
            num_attn_blocks=2,
            num_classes=2, 
            contextlength=200
        )
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

    def run(self):
        args = self.args
        nb_batches = len(self.trainloader)
        for epoch in range(args.epochs):
            self.model.train()
            losses = 0.0
            train_acc = 0.0
            for data, time, label, target in self.trainloader:
                data = data.float().to(self.device)
                time = time.float().to(self.device)
                label = label.float().to(self.device)
                target = target.long().to(self.device)

                out = self.model(data, label, time)
                loss = self.criterion(out, target)

                self.optimizer.zero_grad()
                loss.backward()
                losses += loss.item()
                self.optimizer.step()
                train_acc += (out.argmax(1) == target).detach().cpu().numpy().mean()
                self.scheduler.step()
            
            if args.verbose and (epoch+1) % 10 == 0:
                info = {
                    "epoch" : epoch + 1,
                    "loss" : losses/nb_batches,
                    "train_acc" : train_acc/nb_batches
                }
                print(info)

    def evaluate(self):
        

def main():
    args = SetParams({
        "N": 20,
        "t": 1000,
        "contextlength": 200,
        "seed": 1996,
        "image_size": 28,
        "device": "cpu",
        "lr": 1e-3,
        "epochs": 100,
        "verbose": True
    })
    Trainer(args).run()


    

if __name__ == "__main__":
    main()