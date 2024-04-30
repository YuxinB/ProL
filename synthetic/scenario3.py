import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.auto as tqdm


class Data_Scenario3():
    def __init__(self, p=0.9, τ=30):
        self.p = [[p, 1-p], [1-p, p]]
        self.τ = τ

    def get_samples(self, t):
        y0 = np.random.choice([0, 1])
        y_seq = [y0]
        for _ in range(self.τ-1):
            y_seq.append(np.random.choice([0, 1], p=self.p[y_seq[-1]]))
        y_train, y_test = y_seq[:t], y_seq[t:]
        return y_train, y_test
    
    def evaluate(self, p_hat, y_test):
        p_hat = np.array(p_hat)
        y_hat = np.zeros_like(y_test)
        y_hat[p_hat > 0.5] = 1
        y_hat[ np.abs(p_hat - 0.5) < 1e-5] = 0.5
        err = np.mean(np.abs(p_hat - y_test))
        return err

    def estimator_erm(self, samples):
        p_hat = np.mean(samples)
        ntest = self.τ - len(samples)
        return np.ones(ntest) * p_hat

    def estimator_prospective(self, samples):

        counts = np.array([[1, 1], [1, 1]])
        for i in range(len(samples)-1):
            counts[samples[i], samples[i+1]] += 1
        probs = counts / np.sum(counts, axis=1, keepdims=True)
        p_hat = []
        ycur = samples[-1]
        for t in range(self.τ - len(samples)):
            ycur = np.argmax(probs[ycur])
            p_hat.append(ycur)
        return p_hat
    
    def get_errs(self, seeds, est=0):
        errs = []
        for t in tqdm.tqdm(range(1, self.τ-1)):
            err_seed = []

            for sd in range(seeds):
                y_train, y_test = dat.get_samples(t)

                if est == 0:
                    p_hat = dat.estimator_erm(y_train)
                else:
                    p_hat = dat.estimator_prospective(y_train)
                err_seed.append(dat.evaluate(p_hat, y_test))
            errs.append(err_seed)
        errs = np.array(errs)
        return errs


if __name__ == "__main__":
    p=0.1
    seeds = 1000
    max_t = 100
    times = np.arange(1, max_t-1)

    dat = Data_Scenario3(p, max_t)
    errs_erm = dat.get_errs(seeds, 0)
    errs_prospective = dat.get_errs(seeds, 1)


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

    # plt.plot(times, np.ones_like(times) * bayes_err, '--', label='Bayes Optimal', color='black')
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
    plt.savefig("scenario3.pdf", bbox_inches='tight')
