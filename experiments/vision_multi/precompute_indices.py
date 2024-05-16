from prol.process import (
    get_torch_dataset,
    get_multi_indices_and_map,
    get_multi_sequence_indices
)
from datetime import datetime
import pickle

# specify the task and the experimental details
dataset = 'mnist'
tasks = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4]
]
N = 10
t_list = [100,200,500,700,1000,1200,1500,1700,2000,2500]
T = 5000
initial_seed = 1996
outer_reps = 3
reps = 100

# get the torch dataset
root = '/cis/home/adesilva/ashwin/research/ProL/data'
torch_dataset = get_torch_dataset(root, dataset)

# get the task index dict, label mapper, and updated torch dataset
taskInd, maplab, torch_dataset = get_multi_indices_and_map(tasks, torch_dataset)

# obtain the train/test sequences for the experiment
total_indices = {}
for t in t_list:
    print(f'computing for...t = {t}')
    replicates = []
    for outer_rep in range(outer_reps):
        seed = initial_seed * outer_rep * 2357
        train_SeqInd, updated_taskInd = get_multi_sequence_indices(
            N=N, 
            total_time_steps=t, 
            tasklib=taskInd, 
            seed=seed,
            remove_train_samples=True
        )

        test_seqInds = [
            get_multi_sequence_indices(N, T, updated_taskInd, seed=seed+1000*(inner_rep+1))
            for inner_rep in range(reps)
        ]
        seq = {}
        seq['train'] = train_SeqInd
        seq['test'] = test_seqInds
        replicates.append(seq)
    total_indices[t] = replicates

# save the indices
filename = f'{dataset}_{datetime.now().strftime("%H-%M-%S")}'
file = f'/cis/home/adesilva/ashwin/research/ProL/experiments/vision_multi/indices/{filename}.pkl'
with open(file, 'wb') as f:
    pickle.dump(total_indices, f)