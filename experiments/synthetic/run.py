'''
Boilerplate code for the exps
'''

import importlib
import numpy as np
from tqdm.auto import tqdm
import pickle

import hydra
import logging

from prol.process import get_synthetic_data
from prol.utils import get_dataloader

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
    # input parameters
    params = {
        # experiment params
        "dataset": "synthetic",
        "method": "proformer",
        "N": 20,                     # time between two task switches                   
        "t": cfg.t,                  # training time
        "T": 5000,                   # future time horizon
        "seed": 1996,
        "device": "cuda:1",
        "reps": 100,                 # number of test reps
        "outer_reps": 3,

        # transformer params
        "contextlength": 200, 
        "encoding_type": 'vanilla',
        "multihop": False,

        # training params             
        "lr": 1e-3,         
        "batchsize": 128,
        "epochs": 500,
        "verbose": True
    }
    args = SetParams(params)
    log.info(f'{params}')

    risk_list = []
    for outer_rep in range(args.outer_reps):
        log.info(" ")
        
        # get a training sequence
        seed = args.seed * outer_rep * 2357
        x_train, y_train = get_synthetic_data(
            N=args.N,
            total_time_steps=args.t,
            seed=seed
        )

        # sample a bunch of test sequences
        test_data = [
            get_synthetic_data(args.N, args.T, seed=seed+1000*(inner_rep+1))
            for inner_rep in range(args.reps)
        ]

        # get the module for the specified method
        method, datahandler = get_modules(args.method)

        # form the train dataset
        train_dataset = datahandler.SyntheticSequentialDataset(args, x_train, y_train)

        # model
        model_kwargs = method.model_defaults(args.dataset)
        if args.method == 'proformer':
            model_kwargs['encoding_type'] = args.encoding_type
        log.info(f'{model_kwargs}')
        model = method.Model(
            num_classes=2,
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
            x_test, y_test = test_data[i]
            test_dataset = datahandler.SyntheticSequentialTestDataset(args, x_train, y_train, x_test, y_test)
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
    log.info(f"risk at t = {args.t} : {risk:.4f}")
    
    outputs = {
        "args": params,
        "risk": risk,
        "inst_risk": instantaneous_risk,
        "ci": ci
    }
    with open('outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    main()