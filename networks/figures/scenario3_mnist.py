import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


def online_learners_prospective2():
    fnames = [
        "../checkpoints/mnist_s2/prospective_mlp_l_errs.pkl",
        "../checkpoints/mnist_s2/mlp_l_errs.pkl",
        "../checkpoints/mnist_s2/mlp_ft1_errs.pkl",
        "../checkpoints/mnist_s2/mlp_bgd_errs.pkl",
              ]

    mats = []
    for fname in fnames:
        with open(fname, "rb") as fp:
            info = pickle.load(fp)

        mat = []
        for i in range(len(info)):

            tstep = info[i][0] / 10
            err = info[i][1]
            err = np.mean(err, axis=1)

            err_mean = err.mean()
            err_std = 2 * (np.std(err) / len(err))

            mat.append((tstep, err_mean, err_std))
        mats.append(np.array(mat))

    import ipdb; ipdb.set_trace()

    # vertical line at t=1000
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.75,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    plt.clf()
    plt.figure(figsize=(5, 5))
    plt.ylim([0, 1])
    plt.title("Scenario 2: MNIST")

    plt.plot(mats[0][:, 0], mats[0][:, 1], c='C0')
    plt.plot(mats[1][:, 0], mats[1][:, 1], c='C1')
    plt.plot(mats[2][:, 0], mats[2][:, 1], c='C2')
    plt.plot(mats[3][:, 0], mats[3][:, 1], c='C3')

    plt.scatter(mats[0][:, 0], mats[0][:, 1], label="Time-MLP", s=22, alpha=0.7, c='C0')
    plt.scatter(mats[1][:, 0], mats[1][:, 1], label="Follow-the-leader", s=22, alpha=0.7, c='C1')
    plt.scatter(mats[2][:, 0], mats[2][:, 1], label="Fine-tuning", s=22, alpha=0.7, c='C2')
    plt.scatter(mats[3][:, 0], mats[3][:, 1], label="Bayesian gradient descent", s=22, alpha=0.7, c='C3')

    plt.ylabel("Prospective Risk")
    plt.xlabel("Time (t)")
    plt.legend(markerscale=2., scatterpoints=1, fontsize=15,
               frameon=True)

    plt.savefig("figs/scenario2_mnist_pr2.pdf", bbox_inches='tight')

online_learners_prospective2()
