import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Data_Scenario2():
    def __init__(self, p, times, alpha=5, beta=10, bayes=False):
        self.p = p 
        self.bayes = bayes
        self.alpha = alpha
        self.beta = beta
        self.times = times

    def get_samples(self, t):
        samples = np.random.uniform(0, 1, t)
        # Alterate between using p[0] and p[1] for alternate samples
        parr = np.array([self.p[0] if i % 2 == 0 else self.p[1] for i in range(t)])
        y_true = np.array(samples < parr, dtype=int)
        return y_true

    def evaluate(self, p_hat, t):
        err = 0
        for i in range(2):

            if self.bayes:
                ph = (p_hat[i] * t + self.alpha - 1) / (t + self.alpha + self.beta - 2)
            else:
                ph = p_hat[i]

            if np.abs(ph - 0.5) < 1e-5:
                err +=  0.5 * (1 - self.p[i]) + 0.5 * self.p[i]
            else:
                if ph >= 0.5:
                    err += (1 - self.p[i])
                else:
                    err += (self.p[i])
        err /= 2
        return err

    def estimator_prospective(self, samples):

        p_hat1 = np.mean(samples[::2])
        p_hat2 = np.mean(samples[1::2])

        return [p_hat1, p_hat2]

    def estimator_erm(self, samples):
        p_hat = np.mean(samples)
        return [p_hat, p_hat]
    
    def get_errs(self, seeds, est=0):
        errs = []
        for t in self.times:
            err_seed = []
            for sd in range(seeds):
                samples = dat.get_samples(t)

                if est == 0:
                    p_hat = dat.estimator_erm(samples)
                else:
                    p_hat = dat.estimator_prospective(samples)

                err_seed.append(dat.evaluate(p_hat, t))
            errs.append(err_seed)
        errs = np.array(errs)
        return errs


def bayes_risk_calc(p):
    err = 0
    for pi in p:
        if pi >= 0.5:
            err += 1 - pi
        else:
            err += pi
    return err / 2


if __name__ == "__main__":

    p=[0.9, 0.1]
    seeds = 10000
    times = list(range(2, 20))

    dat = Data_Scenario2(p, times)
    errs_erm = dat.get_errs(seeds, 0)
    errs_prospective = dat.get_errs(seeds, 1)

    bayes_err = bayes_risk_calc(p)


    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.65,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    plt.figure(figsize=(7, 5))

    avg_erm = np.mean(errs_erm, axis=1)
    avg_pr = np.mean(errs_prospective, axis=1)
    std_erm = np.std(errs_erm, axis=1)
    std_pr = np.std(errs_prospective, axis=1)

    plt.plot(times, avg_erm)
    plt.plot(times, avg_pr)

    plt.plot(times, np.ones_like(times) * bayes_err, '--', label='Bayes Optimal', color='black')
    plt.fill_between(times,
                     avg_erm - std_erm / np.sqrt(seeds),
                     avg_erm + std_erm / np.sqrt(seeds), alpha=0.2)
    plt.fill_between(times,
                     avg_pr - std_pr / np.sqrt(seeds),
                     avg_pr + std_pr / np.sqrt(seeds), alpha=0.2)
    plt.ylim([0, 1])


    plt.xlabel("Number of samples / Time (t)")
    plt.ylabel("Average Prospective risk")
    plt.legend(['Maximum likelihood estimator',
                'Time-aware empirical risk minimization',
               ])
    plt.savefig("scenario2.pdf", bbox_inches='tight')

    plt.show()
