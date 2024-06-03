import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'size':16}
matplotlib.rc('font', **font)
matplotlib.rcParams['figure.facecolor'] = 'white'
import seaborn as sns
import numpy as np

# %%

def hypothesis(p, size):
    h = np.random.uniform(low=0, high=1, size=size)
    h[p > 0.5, :] = 1
    h[p < 0.5, :] = 0
    return h

def loss(truth, pred):
    return (truth - pred)**2

def compute_mle(data):
    return np.mean(data, axis=-1)

def compute_map(self, data):
    nom = self.alpha + np.sum(data, axis=-1) - 1
    denom = self.alpha + self.beta + self.t - 2
    return nom/denom

# %%

p = 0.8
T = 20
T_end = 1000
reps = 10000

# %%

# sample a bunch of sequences from the process
outcomes = np.random.binomial(1, p, size=(reps, T_end))

t_list = np.arange(1, T, 1)
risk = []

# loop over increasing values of t
for t in t_list:

    # get the past (training) data at each outcome
    past_data = outcomes[:, :t]

    # get the future (evaluation) data at each outcome
    future_data = outcomes[:, t:]

    # compute the p_hat at each outcome based on past data
    p_hat = compute_mle(past_data)
    
    # get the hypothesis for each outcome
    h = hypothesis(p_hat, future_data.shape)

    # get the cumulative loss for each outcome
    cumulative_loss = np.mean(loss(future_data, h), axis=-1)

    # take the weighted average of the cumulative loss over all the outcomes (weight is the probabilities we computed above)
    expected_cumulative_loss = np.mean(cumulative_loss)
    risk.append(expected_cumulative_loss)

# %%

fig, ax = plt.subplots()
color = sns.color_palette("tab10")[3]
ax.plot(t_list, risk, label='MLE', marker='o', ms=5, lw=2)
ax.plot(t_list, (1-p)*np.ones((len(t_list),)), label='Bayes risk', color='k', ls='dashed', lw=2)
ax.legend(frameon=False)
ax.set_ylabel('prospective risk')
ax.set_xlabel(r'number of samples / time ($t$)')
ax.set_title('independent and identically distributed', fontsize=16, fontweight='bold', y=1.05)
ax.set_ylim([0, 1])
plt.show()

# %%

case1_outputs = {
    'risk': risk,
    'time': t_list
}

file = 'results/scenario1.npy'
np.save(file, case1_outputs, allow_pickle=True)

# %%

fig.savefig("results/case1.pdf", bbox_inches='tight')