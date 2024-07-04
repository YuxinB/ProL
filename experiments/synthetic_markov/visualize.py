# %%
import numpy as np
import pickle
import matplotlib as mpl
font = {'size':16}
mpl.rc('font', **font)
mpl.rcParams['figure.facecolor'] = 'white'
import matplotlib.pyplot as plt

def compute_discounted_prosp_risk(outputs, gamma=1):
    preds = np.array(outputs['raw_metrics']['preds'])
    truths = np.array(outputs['raw_metrics']['truths'])
    shape = preds.shape
    undiscounted_loss = (preds != truths).astype('int')
    weights = np.power(gamma, np.arange(0, shape[2]))
    discounted_loss = np.sum(undiscounted_loss * weights, axis=-1)
    discounted_risks = np.mean(discounted_loss, axis=-1)
    return (1-gamma)*np.mean(discounted_risks)

# %%
gamma = 0.99
eps_list = [0.001, 0.01, 0.1]
file_list = [
    'multirun/2024-06-10/04-43-21',
    'multirun/2024-06-10/04-43-22',
    'multirun/2024-06-10/04-43-25'
]

fig, ax = plt.subplots(figsize=(10, 7))

for i, eps in enumerate(eps_list):
    time_list = []
    risk_list = []
    ci_list = []
    file = file_list[i]
    for j in range(10):
        fname = f'{file}/{j}/outputs.pkl'
        with open(fname, 'rb') as f:
            outputs = pickle.load(f)
        time_list.append(outputs['args']['t'])
        risk_list.append(compute_discounted_prosp_risk(outputs, gamma))
        # risk_list.append(outputs['risk'])
    risks = np.array(risk_list)

    ax.plot(time_list, risks, lw=3, label=r'$\epsilon = $' + f'{eps}')

ax.legend()
ax.set_ylim([0, 1])
ax.set_ylabel('Prospective Risk')
ax.set_xlabel('Time')
plt.show()

# %%

fname = f'synthetic_3.pkl'
with open(fname, 'rb') as f:
    outputs = pickle.load(f)

# %%
