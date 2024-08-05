import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


def retro_vs_prospective():
    """
    Compare retrospective and prospective MLP algorithms
    """
    fnames = ["../checkpoints/scenario3/mlp_l_errs.pkl",
              "../checkpoints/scenario3/prospective_mlp_l_errs.pkl"]

    mats = []
    for fname in fnames:
        with open(fname, "rb") as fp:
            info = pickle.load(fp)

        mat = []
        for i in range(len(info)):

            tstep = info[i][0]
            err = info[i][1]
            err = np.mean(err, axis=1)

            err_mean = err.mean()
            err_std = 2 * (np.std(err) / len(err))

            mat.append((tstep, err_mean, err_std))
        mats.append(mat)

    mats = np.array(mats)
    plt.style.use("seaborn-v0_8-whitegrid")

    sns.set(context='poster',
            style='ticks',
            font_scale=0.85,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    for i in range(2):
        plt.plot(mats[i][:, 0], mats[i][:, 1])

    # horizontal line
    plt.axhline(y=0.0, color='black', linestyle='--')

    cols = ['b', 'orange']

    for i in range(2):
        plt.scatter(mats[i][:, 0], mats[i][:, 1], color=cols[i])

        plt.fill_between(mats[i][:, 0],
                         mats[i][:, 1] - mats[i][:, 2],
                         mats[i][:, 1] + mats[i][:, 2],
                         alpha=0.5, color=cols[i])

    plt.legend(["Prospective-MLP", "Retrospective-MLP", "Bayes-risk"])

    plt.title("Scenario 3: Synthetic Data")
    plt.xlabel("Time (t)")
    plt.ylabel("Prospective Risk")

    plt.savefig("figs/scenario3_syn.pdf", bbox_inches='tight')

retro_vs_prospective()

