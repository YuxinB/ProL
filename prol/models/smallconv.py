import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm

# define the base CNN
class Model(nn.Module):
    """
    Small convolution network with no residual connections (single-head)
    """
    def __init__(self, num_classes=10, channels=3, avg_pool=2, lin_size=320):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(channels, 80, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(avg_pool)

        self.linsize = lin_size
        self.fc = nn.Linear(self.linsize, num_classes)

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
    
def model_defaults(dataset):
    if dataset == 'mnist':
        return {
            "channels": 1,
            "avg_pool": 2, 
            "lin_size": 80
        }
    elif dataset == 'cifar-10':
        return {
            "channels": 3,
            "avg_pool": 2, 
            "lin_size": 320
        }
    else:
        raise NotImplementedError

class SequentialDataset(Dataset):
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
        """
        self.args = args
        self.dataset = dataset
        self.seqInd = seqInd
        self.maplab = maplab

    def __len__(self):
        return len(self.seqInd)

    def __getitem__(self, idx):
        data = self.dataset.data[self.seqInd[idx]][None, :]
        label = self.dataset.targets[self.seqInd[idx]].apply_(self.maplab)
        return data, label

class SequentialTestDataset(Dataset):
    def __init__(self, args, dataset, train_seqInd, test_seqInd, maplab) -> None:
        t = len(train_seqInd)
        self.dataset = dataset
        self.test_seqInd = test_seqInd[t:]
        self.maplab = maplab
        
    def __len__(self):
        return len(self.test_seqInd)
        
    def __getitem__(self, idx):
        data = self.dataset.data[self.seqInd[idx]][None, :]
        label = self.dataset.targets[self.test_seqInd[idx]].apply_(self.maplab)
        return data, label
    
class Trainer:
    def __init__(self, model, dataset, args) -> None:
        self.args = args

        self.trainloader = DataLoader(dataset, batch_size=args.batchsize)

        self.model = model
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, log):
        args = self.args
        nb_batches = len(self.trainloader)
        for epoch in range(args.epochs):
            self.model.train()
            losses = 0.0
            train_acc = 0.0
            for data, target in self.trainloader:
                data = data.float().to(self.device)
                target = target.long().to(self.device)

                out = self.model(data)
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
                    "loss" : np.round(losses/nb_batches, 4),
                    "train_acc" : np.round(train_acc/nb_batches, 4)
                }
                print(info)
                log.info(f'{info}')

    def evaluate(self, testloader, verbose=False):
        self.model.eval()
        preds = []
        truths = []
        if verbose:
            progress = tqdm(testloader)
        else:
            progress = testloader
        for data, target in progress:
            data = data.float().to(self.device)
            target = target.long().to(self.device)

            out = self.model(data)

            preds.extend(
                out.detach().cpu().argmax(1).numpy()
            )
            truths.extend(
                target.detach().cpu().numpy()
            )
        return np.array(preds), np.array(truths)
    
def main():
    # testing
    x = torch.randn(1, 1, 28, 28)
    net = Model(10, 1, 2, 80)
    y = net(x)
    print(y.shape)

    x = torch.randn(1, 3, 32, 32)
    model_kwargs = model_defaults(dataset='cifar-10')
    net = Model(2, **model_kwargs)
    y = net(x)
    print(y.shape)


if __name__ == "__main__":
    main()