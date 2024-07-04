# %%
'''
Precompute task seuqnece for synthetic Markov (case 3) exps
'''

from prol.process import (
    get_markov_chain
)
from datetime import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt

# %%
# specify the task and the experimental details
dataset = 'synthetic'
N = 20
T = 5000
initial_seed = 1996
outer_reps = 3
num_tasks = 2
eps_list = [0.0001, 0.001, 0.01, 0.1]

# %% 
outputs = {}
for i, eps in enumerate(eps_list):
    # transition matrix
    P = np.array([
        [eps, 1-eps],
        [1-eps, eps]
    ])
    patterns = np.array([get_markov_chain(num_tasks, P, T, N, seed=k*11111) for k in range(outer_reps)])
    outputs[eps] = patterns

outputs["N"] = N
outputs["T"] = T
outputs["outer_reps"] = outer_reps
outputs["eps_list"] = eps_list

fname = f'{dataset}_{outer_reps}.pkl'
with open(fname, 'wb') as f:
    pickle.dump(outputs, f)



