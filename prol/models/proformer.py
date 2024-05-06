import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, p=0.1):
        super().__init__()
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim, p),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model, p),
        )
        self.mha = nn.MultiheadAttention(d_model, num_heads, p)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ff_output = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + ff_output)
        return out2

class Model(nn.Module):
    def __init__(self, input_size, d_model, num_heads, ff_hidden_dim, num_attn_blocks=1, num_classes=2, 
                 contextlength=200, max_len=5000):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        self.attention_blocks = nn.ModuleList(
            [SelfAttention(d_model, num_heads, ff_hidden_dim) for _ in range(num_attn_blocks)]
        )
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(contextlength+1, contextlength+1, kernel_size=3, stride=2)
        )
        self.input_embedding = nn.Linear(input_size+1, d_model//2)
        self.layernorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.classifier = nn.Linear(d_model, num_classes)

        # frequency-adjusted fourier encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = 2 * math.pi / torch.arange(2, d_model//2 + 1, 2)
        ffe = torch.zeros(1, max_len, d_model//2)
        ffe[0, :, 0::2] = torch.sin(position * div_term)
        ffe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('ffe', ffe)
    
    def time_encoder(self, t):
        enc = torch.cat([self.ffe[:, t[i].squeeze().long(), :] for i in range(t.size(0))])
        return enc
        
    def forward(self, data, labels, times):
        # v = self.conv_encoder(data)  
        # v = v.flatten(-2, -1)      
        u = torch.cat((data, labels.unsqueeze(-1)), dim=-1)
        u = self.input_embedding(u)

        t = self.time_encoder(times)

        x = torch.cat((u, t), dim=-1)

        for attn_block in self.attention_blocks:
            x = attn_block(x)
        x = torch.select(x, 1, -1)
        x = self.classifier(x)
        return x
    
def model_defaults():
    return {
        "input_size": 28*28,
        "d_model": 512, 
        "num_heads": 8,
        "ff_hidden_dim": 2048,
        "num_attn_blocks": 4,
        "contextlength": 200
    }  
    
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
        contextlength : int, optional
            length of the history
        """
        self.dataset = dataset
        self.contextlength = args.contextlength
        self.t = len(seqInd)
        self.time = torch.arange(self.t).float()
        self.seqInd = seqInd
        self.maplab = maplab

    def __len__(self):
        return len(self.seqInd)

    def __getitem__(self, idx):
        r = np.random.randint(0, len(self.seqInd)-2*self.contextlength) # select the start of the history
        s = np.random.randint(r+self.contextlength, r+2*self.contextlength)  # select a 'future' datum

        id = list(range(r, r+self.contextlength)) + [s]
        dataid = self.seqInd[id] # get indices for the context window

        data = self.dataset.data[dataid]
        data = data.view(data.shape[0], data.shape[-1]**2)
        labels = self.dataset.targets[dataid].apply_(self.maplab)
        time = self.time[id]
        
        target = labels[-1].clone() # true label of the future datum
        labels[-1] = np.random.binomial(1, 0.5) # replace the true label of the future datum with a random label

        return data, time, labels, target
    
class SequentialTestDataset(Dataset):
    def __init__(self, args, dataset, train_seqInd, test_seqInd, maplab) -> None:
        t = len(train_seqInd)
        self.dataset = dataset
        self.contextlength = args.contextlength
        self.train_seqInd = train_seqInd[-args.contextlength:]
        self.test_seqInd = test_seqInd[t:]
        self.maplab = maplab
        
        self.train_time = torch.arange(t).float()
        self.test_time = torch.arange(t, t + len(test_seqInd)).float()
        
    def __len__(self):
        return len(self.test_seqInd)
        
    def __getitem__(self, idx):
        dataid = self.train_seqInd.tolist() + [self.test_seqInd[idx]] # most recent history + inference datum indices

        data = self.dataset.data[dataid]
        data = data.view(data.shape[0], data.shape[-1]**2)
        labels = self.dataset.targets[dataid].apply_(self.maplab)
        time = torch.cat([
            self.train_time[-self.contextlength:], 
            self.test_time[idx].view(1)
        ])

        target = labels[-1].clone() # true label of the future datum
        labels[-1] = np.random.binomial(1, 0.5) # replace the true label of the future datum with a random label

        return data, time, labels, target

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
        for data, time, label, target in progress:
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

if __name__ == "__main__":
    # testing
    kwargs = model_defaults()
    net = Model(num_classes=2, **kwargs)
    data = torch.randn((64, 201, 28, 28))
    labels = torch.randn((64, 201))
    times = torch.randn((64, 201))
    y = net(data, labels, times)
    print(y.shape)