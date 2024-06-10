import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Data_Scenario1():
    def __init__(self, p=0.5, alpha=1, beta=1, bayes=False):
        self.p = 0.9
        self.bayes = bayes
        self.alpha = alpha
        self.beta = beta

    def get_samples(self, t):
        samples = np.random.uniform(0, 1, t)
        y_true = np.array(samples < self.p, dtype=int)
        return y_true
    
    def evaluate(self, p_hat, t):

        if self.bayes:
            p_hat = (p_hat * t + self.alpha - 1) / (t + self.alpha + self.beta - 2)


        if np.abs(p_hat - 0.5) < 1e-5:
            return 0.5 * (1 - self.p) + 0.5 * self.p
        else:
            if p_hat >= 0.5:
                err = 1 - self.p
            else:
                err = self.p
        return err
    
    def get_errs(self, seeds):
        errs = []
        for t in times:
            err_seed = []
            for sd in range(seeds):
                samples = dat.get_samples(t)
                p_hat = np.mean(samples)
                err_seed.append(dat.evaluate(p_hat, t))
            errs.append(err_seed)
        errs = np.array(errs)
        return errs


p=0.9
seeds = 10000
times = list(range(1, 20))

dat = Data_Scenario1(p)
errs = dat.get_errs(seeds)

dat = Data_Scenario1(p, alpha=2, beta=4, bayes=True)
errs_b = dat.get_errs(seeds)

avg_errs = np.mean(errs, axis=1)
std_errs = np.std(errs, axis=1) / np.sqrt(seeds)

avg_errs_b = np.mean(errs_b, axis=1)
std_errs_b = np.std(errs_b, axis=1) / np.sqrt(seeds)

info = {
        'avg_errs': avg_errs,
        'std_errs': std_errs,
}
np.save('scenario1.npy', info, allow_pickle=True)
    

plt.style.use("seaborn-v0_8-whitegrid")
sns.set(context='poster',
        style='ticks',
        font_scale=0.65,
        rc={'axes.grid':True,
            'grid.color':'.9',
            'grid.linewidth':0.75})

plt.figure(figsize=(7, 5))

plt.plot(times, avg_errs)
plt.plot(times, avg_errs_b)
plt.plot(times, np.ones_like(times) * (1-p), '--', label='Bayes Optimal', color='black')
plt.fill_between(times, avg_errs - std_errs, avg_errs + std_errs, alpha=0.2)
plt.fill_between(times, avg_errs_b - std_errs_b, avg_errs_b + std_errs_b, alpha=0.2)
plt.ylim([0, 1])
plt.title('Independent and identically distributed')


plt.xlabel("Number of samples / Time (t)")
plt.ylabel("Average Prospective risk")
plt.legend(['Maximum likelihood estimator',
            'Mode of Bayes Posterior \n(alpha=2, beta=4)',
            'Bayes Optimal'])

# plt.savefig("scenario1.pdf", bbox_inches='tight')

