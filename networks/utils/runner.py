import torch
import os
import torch.nn as nn



def train(cfg, net, loaders):

    dev = cfg.dev
    trainloader, testloader = loaders
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01,
                                momentum=0.9, nesterov=True,
                                weight_decay=0.00001)

    net.train()

    for ep in range(cfg.train.epochs):
        for dat, targets, time in trainloader:
            dat, targets = dat.to(dev), targets.to(dev)
            logits = net(dat, time)

            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: %d, Loss: %.4f" % (ep, loss.item()))

    import ipdb; ipdb.set_trace()
    net.eval()
    save_net(net, cfg)


def save_net(net, cfg):
    info = {
        'state_dict': net.state_dict(),
        'cfg': cfg
    }
    fpath = os.path.join('checkpoints', cfg.tag)
    os.makedirs(fpath, exist_ok=True)
    torch.save(info, os.path.join(fpath, cfg.name))


def log_train():
    pass


def log_eval():
    pass
