import torch
from torch.utils.data import DataLoader
import numpy as np

def wif(id):
    """
    Used to fix randomization bug for pytorch dataloader + numpy
    Code from https://github.com/pytorch/pytorch/issues/5059
    """
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

def get_dataloader(dataset, batchsize, train=True):
    loader_kwargs = {
            'worker_init_fn': wif,
            'pin_memory': True,
            'num_workers': 4
    }
    loader = DataLoader(
                dataset, 
                batch_size=batchsize,
                shuffle=train,
                **loader_kwargs
            )
    return loader
