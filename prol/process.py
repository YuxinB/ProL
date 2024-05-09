import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
from copy import deepcopy

def get_cycle(N: int) -> list:
    """Get the primary cycle

    Parameters
    ----------
    N : int
        time between two task switches

    Returns
    -------
    list
        primary cycle
    """
    return [1] * N + [0] * N

def get_torch_dataset(root, name='mnist'):
    """Get the original torch datase

    Returns
    -------
    _type_
        torch dataset
    """
    if name == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root=root,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.Lambda(lambda x : torch.flatten(x))
            ]),
            download=True
        )
        # normalize
        tmp = dataset.data.float() / 255.0
        tmp = (tmp - 0.5)/0.5
        dataset.data = tmp[..., None]

    elif name == 'cifar-10':
        dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]),
            download=True
        )
        # normalize
        tmp = torch.from_numpy(dataset.data).float() / 255.0
        tmp = (tmp - 0.5)/0.5
        dataset.data = tmp
        tmp = dataset.targets
        dataset.targets = torch.Tensor(tmp).long()

    assert dataset.data.max() == 1.0
    assert dataset.data.min() == -1.0
    return dataset

def get_task_indicies_and_map(tasks: list, y: np.ndarray):
    """Get the indices for each task + the label mapping

    Parameters
    ----------
    tasks : list
        task specification e.g. [[0, 1], [2, 3]]
    y : np.ndarray
        dataset targets
    """
    tasklib = {}
    for i, task in enumerate(tasks):
        tasklib[i] = []
        for lab in task:
            tasklib[i].extend(
                np.where(y == lab)[0].tolist()
            )
    mapdict = {}
    for task in tasks:
        for i, lab in enumerate(task):
            mapdict[lab] = i
    maplab = lambda lab : mapdict[lab]
    return tasklib, maplab

def get_sequence_indices(N, total_time_steps, tasklib, seed=1996, remove_train_samples=False):
    """Get indices for a sequence drawn from the stochastic process

    Parameters
    ----------
    N : time between two task switches
    total_time_steps : length of the sequence drawn
    tasklib : original task indices
    seed : random seed

    Returns
    -------
    index sequence
    """
    tasklib = deepcopy(tasklib)
    unit = get_cycle(N)
    pattern = np.array((unit * math.ceil(total_time_steps/(len(unit))))[:total_time_steps]).astype("bool")
    seqInd = np.zeros((total_time_steps,)).astype('int')
    np.random.seed(seed)
    seqInd[pattern] = np.random.choice(tasklib[0], sum(pattern), replace=False)
    seqInd[~pattern] = np.random.choice(tasklib[1], sum(~pattern), replace=False)

    if remove_train_samples:
        tasklib[0] = list(
            set(tasklib[0]) - set(tasklib[0]).intersection(seqInd)
        )
        tasklib[1] = list(
            set(tasklib[1]) - set(tasklib[1]).intersection(seqInd)
        )
        return seqInd, tasklib
    else:
        return seqInd

def draw_synthetic_samples(flip):
    y = np.random.binomial(1, 0.5)
    l = 1-y if flip else y
    x = l * np.random.uniform(-2, -1) + (1-l) * np.random.uniform(1, 2)
    return x, y

def get_synthetic_data(N, total_time_steps, seed=1996):
    """Get synthetic data sequence drawn from the stochastic process

    Parameters
    ----------
    N : time between two task switches
    total_time_steps : length of the sequence drawn
    seed : random seed

    Returns
    -------
    index sequence
    """
    unit = get_cycle(N)
    pattern = np.array((unit * math.ceil(total_time_steps/(len(unit))))[:total_time_steps]).astype("bool")
    data = np.zeros((total_time_steps, 2)).astype('float')
    np.random.seed(seed)
    data[pattern] = np.array([draw_synthetic_samples(True) for _ in range(sum(pattern))])
    data[~pattern] = np.array([draw_synthetic_samples(False) for _ in range(sum(~pattern))])
    return torch.from_numpy(data[:, 0]).float(), torch.from_numpy(data[:, 1]).long()

if __name__ == "__main__":
    x, y = get_synthetic_data(20, 100)
    print(x.shape)

