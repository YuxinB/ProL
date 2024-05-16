'''
Vision covariate shift exps
'''
import importlib
import torch
import numpy as np
from tqdm.auto import tqdm
import pickle

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import logging

from prol.process import (
    get_torch_dataset,
    get_multi_indices_and_map
)

import pathlib

class SetParams:
    def __init__(self, dict) -> None:
        for k, v in dict.items():
            setattr(self, k, v)

def get_modules(name):
    try: 
        module1 = importlib.import_module(f"prol.models.{name}")
        module2 = importlib.import_module(f"prol.datahandlers.{name}_handle")
    except ImportError:
        print(f"Module {name} not found")
    return module1, module2

log = logging.getLogger(__name__)

@hydra.main(config_path=".", config_name="config")
def main(cfg):
    cwd = pathlib.Path(get_original_cwd())

    # input parameters
    params = {
        # dataset
        "dataset": "mnist",
        "task": [[0, 1, 2], [1, 2, 3], [2, 3, 4]],    # task specification
        "indices_file": 'mnist_16-02-38',

        # experiment
        "method": "cnn",         # select from {proformer, cnn, mlp, timecnn}
        "N": 10,                     # time between two task switches                   
        "t": cfg.t,                  # training time
        "T": 5000,                   # future time horizon
        "seed": 1996,   
        "device": "cuda:0",          # device
        "reps": 100,                 # number of test reps
        "outer_reps": 3,         
       
        # proformer
        "proformer" : {
            "contextlength": 200, 
            "encoding_type": 'freq',      
            "multihop": True
        },

        # conv_proformer
        "conv_proformer" : {
            "contextlength": 100, 
            "encoding_type": 'freq',      
            "multihop": True
        },

        # timecnn
        "timecnn": {
            "encoding_type": 'freq', 
        },
              
        # training params
        "lr": 1e-3,         
        "batchsize": 64,
        "epochs": 500,
        "verbose": True
    }
    args = SetParams(params)
    log.info(f'{params}')

    # max number of classes
    max_num_classes = max([len(task) for task in args.task])

    # get source dataset
    root = '/cis/home/adesilva/ashwin/research/ProL/data'
    torch_dataset = get_torch_dataset(root, name=args.dataset)
    
    # get indices for each task
    _, maplab, torch_dataset = get_multi_indices_and_map(
        tasks=args.task,
        dataset=torch_dataset
    )

    # load the saved indicies
    indices_file = cwd / f'indices/{args.indices_file}.pkl'
    with open(indices_file, 'rb') as f:
        total_indices = pickle.load(f)

    risk_list = []
    for outer_rep in range(args.outer_reps):
        log.info(" ")
        
        # get a training sequence
        train_SeqInd = total_indices[args.t][outer_rep]['train']

        # sample a bunch of test sequences
        test_seqInds = total_indices[args.t][outer_rep]['test']

        # get the module for the specified method
        method, datahandler = get_modules(args.method)

        # form the train dataset
        data_kwargs = {
            "dataset": torch_dataset, 
            "seqInd": train_SeqInd, 
            "maplab": maplab
        }
        train_dataset = datahandler.VisionSequentialDataset(args, **data_kwargs)

        # model
        model_kwargs = method.model_defaults(args.dataset)
        if args.method == 'proformer':
            model_kwargs['encoding_type'] = args.proformer['encoding_type']
        elif args.method == 'timecnn':
            model_kwargs['encoding_type'] = args.timecnn['encoding_type']
        log.info(f'{model_kwargs}')
        model = method.Model(
            num_classes=max_num_classes,
            **model_kwargs
        )
        
        # train
        trainer = method.Trainer(model, train_dataset, args)
        trainer.fit(log)

        # evaluate
        preds = []
        truths = []
        for i in tqdm(range(args.reps)):
            # form a test dataset for each test sequence
            test_kwargs = {
                "dataset": torch_dataset, 
                "train_seqInd": train_SeqInd, 
                "test_seqInd": test_seqInds[i], 
                "maplab": maplab
            }
            test_dataset = datahandler.VisionSequentialTestDataset(args, **test_kwargs)
            preds_rep, truths_rep = trainer.evaluate(test_dataset)
            preds.append(preds_rep)
            truths.append(truths_rep)
        preds = np.array(preds)
        truths = np.array(truths)

        # compute metrics
        instantaneous_risk = np.mean(preds != truths, axis=0).squeeze()
        std_error = np.std(preds != truths, axis=0).squeeze()
        ci = std_error * 1.96/np.sqrt(args.reps).squeeze()

        time_averaged_risk = np.mean(preds != truths)
        log.info(f"error = {time_averaged_risk:.4f}")
        risk_list.append(time_averaged_risk)

    risk = np.mean(risk_list)
    ci_risk = np.std(risk_list) * 1.96/np.sqrt(args.outer_reps).squeeze()
    log.info(f"risk at t = {args.t} : {risk:.4f}")
    
    outputs = {
        "args": params,
        "risk": risk,
        "ci_risk": ci_risk, 
        "inst_risk": instantaneous_risk,
        "ci": ci
    }
    with open('outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)

    # save last model
    torch.save(trainer.model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()