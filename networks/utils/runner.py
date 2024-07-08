import torch
import numpy as np
import os
import torch.nn as nn

from utils.bgd import BGD


def train(cfg, net, loaders):

    dev = cfg.dev
    trainloader, testloader = loaders
    criterion = nn.CrossEntropyLoss()

    if cfg.bgd:
        optimizer = BGD(net.parameters(), std_init=0.01)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01,
                                    momentum=0.9, nesterov=True,
                                    weight_decay=0.00001)

    net.train()
    net.to(cfg.dev)

    for ep in range(cfg.train.epochs):
        for dat, targets, time in trainloader:
            dat, targets = dat.to(dev), targets.to(dev)
            time = time.to(dev)

            if cfg.bgd:
                for mc_iter in range(10):
                    optimizer.randomize_weights()
                    logits = net(dat, time)
                    loss = criterion(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.aggregate_grads()
            else:
                logits = net(dat, time)
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()

            optimizer.step()

    print("Epoch: %d, Loss: %.4f" % (ep, loss.item()))
    net.eval()

    errs = evaluate(cfg, net, testloader)
    return errs


def evaluate(cfg, net, testloader):
    dev = cfg.dev
    errs = []
    for dat, targets, time in testloader:
        dat = dat.to(dev)
        targets = targets.to(dev)
        time = time.to(dev)
        logits = net(dat, time)
        probs = torch.softmax(logits, dim=1)
        err = (probs.argmax(dim=1) != targets).float()

        errs.append(err.cpu().numpy())
    errs = np.concatenate(errs)
    return errs


def save_net(net, cfg):
    info = {
        'state_dict': net.state_dict(),
        'cfg': cfg
    }
    fpath = os.path.join('checkpoints', cfg.tag)
    os.makedirs(fpath, exist_ok=True)
    torch.save(info, os.path.join(fpath, cfg.name + ".pth"))


def log_train():
    pass


def log_eval():
    pass
