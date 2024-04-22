import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from copy import deepcopy
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

from prol.models.protransformer import TransformerClassifier

class SetParams:
    def __init__(self, dict) -> None:
        for k, v in dict.items():
            setattr(self, k, v)

def get_pattern(N):
    return [1] * N + [0] * N

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

def get_data_sequence(pattern, dist_A, dist_B, seed=1996):
    data, targets = get_data(dist_A, sum(pattern), 1996)
    seqData_shape = (len(pattern),) + data.shape[1:]
    seqData = np.zeros(seqData_shape)
    seqLab = np.zeros((len(pattern),))

    seqData[pattern], seqLab[pattern] = data, targets
    seqData[~pattern], seqLab[~pattern] = get_data(dist_B, sum(~pattern), 1996)

    seqData = torch.from_numpy(seqData)
    seqLab = torch.from_numpy(seqLab).long()
    return seqData, seqLab

class SequentialDataset(Dataset):
    def __init__(self, N, t, contextlength=200, seed=1996):
        dataset = torchvision.datasets.MNIST(
            root="../data",
            train=True,
            download=True
        )
        dist_A = get_sub_dataset(dataset, (0, 1))
        dist_B = get_sub_dataset(dataset, (2, 3))

        unit = get_pattern(N)
        pattern = np.array((unit * math.ceil(t/(2*N)))[:t]).astype("bool")

        data, self.labels = get_data_sequence(pattern, dist_A, dist_B, seed=seed)
        self.data = data.view(data.shape[0], data.shape[-1]**2)
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
    def __init__(self, N, t, T, train_data, train_labels, contextlength, seed) -> None:
        self.train_data = train_data
        self.train_labels = train_labels
        dataset = torchvision.datasets.MNIST(
            root="../data",
            train=False,
            download=True
        )
        dist_A = get_sub_dataset(dataset, (0, 1))
        dist_B = get_sub_dataset(dataset, (2, 3))

        unit = get_pattern(N)
        pattern = np.array((unit * math.ceil((t+T)/(2*N))))[t:t+T].astype("bool")

        test_data, self.test_labels = get_data_sequence(pattern, dist_A, dist_B, seed=seed)
        self.test_data = test_data.view(test_data.shape[0], test_data.shape[-1]**2)
        self.contextlength = contextlength
        self.t = t
        self.train_time = torch.arange(t).float()
        self.test_time = torch.arange(t, t+T).float()

    def __len__(self):
        return len(self.test_labels)
        
    def __getitem__(self, idx):
        data = torch.cat((self.train_data[-self.contextlength:], self.test_data[idx:idx+1]), axis=0)
        labels = torch.cat((self.train_labels[-self.contextlength:], self.test_labels[idx:idx+1]), axis=0)
        time = torch.cat((self.train_time[-self.contextlength:], self.test_time[idx:idx+1]), axis=0)

        target = labels[-1].clone()
        labels[-1] = np.random.binomial(1, 0.5)

        return data, time, labels, target

class Trainer:
    def __init__(self, args) -> None:
        self.args = args

        self.dataset = SequentialDataset(
            N=args.N,
            t=args.t,
            contextlength=args.contextlength,
            seed=args.seed
        )
        self.trainloader = DataLoader(self.dataset, batch_size=args.batchsize)

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
            
            if args.verbose and (epoch+1) % 1 == 0:
                info = {
                    "epoch" : epoch + 1,
                    "loss" : np.round(losses/nb_batches, 4),
                    "train_acc" : np.round(train_acc/nb_batches, 4)
                }
                print(info)

    def evaluate(self, testloader):
        self.model.eval()
        preds = []
        truths = []
        for data, time, label, target in tqdm(testloader):
            data = data.float().to(self.device)
            time = time.float().to(self.device)
            label = label.float().to(self.device)
            target = target.long().to(self.device)

            out = self.model(data, label, time)

            preds.extend(
                out.detach().cpu().argmax(1).numpy()
            )
            truths.extend(
                target.detach().cpu().numpy()
            )
        return np.array(preds), np.array(truths)
        
def plotting(y, ci, args):
    t = args.t
    T = args.T
    N = args.N
    time = np.arange(t, t+T)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, y, c="k", lw=2)
    ax.fill_between(time, y-ci, y+ci, alpha=0.2, color='k')

    unit = get_pattern(N)
    pattern = np.array((unit * math.ceil((t+T)/(2*N))))[t:t+T].astype("bool")

    for i in time[pattern]:
        ax.add_artist(Rectangle((i, 0), 1, 1, alpha=0.4, edgecolor=None, facecolor="blue"))
    for i in time[~pattern]:
        ax.add_artist(Rectangle((i, 0), 1, 1, alpha=0.4, edgecolor=None, facecolor="orange"))

    ax.set_xlabel("s")
    ax.set_ylabel("risk")

    plt.show()
    plt.savefig("test.png", bbox_inches="tight")

def main():
    args = SetParams({
        "N": 20,
        "t": 1000,
        "T": 1000,
        "contextlength": 200,
        "seed": 1996,
        "image_size": 28,
        "device": "cuda:1",
        "lr": 1e-3,
        "batchsize": 128,
        "epochs": 1,
        "verbose": True,
        "reps": 5
    })

    # train
    trainer = Trainer(args)
    trainer.run()

    # evaluate
    preds = []
    truths = []
    for i in range(args.reps):
        testdataset = SequentialTestDataset(
            args.N, 
            args.t, 
            args.T, 
            trainer.dataset.data, 
            trainer.dataset.labels, 
            contextlength=args.contextlength,
            seed=3000 + 500*i
        )
        testloader = DataLoader(
            testdataset, 
            batch_size = 100,
            shuffle=False
        )
        preds_rep, truths_rep = trainer.evaluate(testloader)
        preds.append(preds_rep)
        truths.append(truths_rep)
    preds = np.array(preds)
    truths = np.array(truths)

    mean_error = np.mean(preds != truths, axis=0).squeeze()
    std_error = np.std(preds != truths, axis=0).squeeze()
    ci = std_error * 1.96/np.sqrt(args.reps).squeeze()

    err = np.mean(preds != truths)
    print(f"error = {err:.4f}")

    plotting(mean_error, ci, args)


if __name__ == "__main__":
    main()