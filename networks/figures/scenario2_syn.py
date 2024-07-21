import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


def retro_vs_prospective():
    """
    Compare retrospective and prospective MLP algorithms
    """
    fnames = ["../checkpoints/scenario2/prospective_mlp_period20_errs.pkl",
              "../checkpoints/scenario2/retrospective_mlp_period20_errs.pkl"]

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

    plt.title("Scenario 2: Synthetic Data")
    plt.xlabel("Time (t)")
    plt.ylabel("Prospective Risk")

    plt.savefig("figs/scenario2_syn.pdf", bbox_inches='tight')


def retro_vs_prospective_instantaneous():
    """
    Risk of 
    """
    fnames = ["../checkpoints/scenario2/prospective_mlp_period20_errs.pkl",
              "../checkpoints/scenario2/retrospective_mlp_period20_errs.pkl",
              ]

    infos = []
    for fname in fnames:
        with open(fname, "rb") as fp:
            info = pickle.load(fp)
        infos.append(info)

    prosp = infos[0][5][1][1]
    retro = infos[1][5][1][1]

    times = np.arange(len(prosp))

    # vertical line at t=1000

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.85,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    plt.subplot(1, 2, 1)
    plt.scatter(times[::50], prosp[::50], label="Prospective-MLP",
                s=2.0, alpha=0.8, marker='x')
    plt.axvline(x=1000, color='black', linestyle='--', lw=1.5)
    plt.ylabel("Instantaneous Risk")
    plt.xlabel("Time (t)")
    plt.legend(loc="center right", markerscale=6., scatterpoints=1, fontsize=10)

    plt.subplot(1, 2, 2)
    plt.scatter(times[::50], retro[::50], label="Retrospective-MLP",
                s=2.0, alpha=0.8, c='orange', marker='x')
    plt.ylabel("Instantaneous Risk")
    plt.xlabel("Time (t)")

    plt.axvline(x=1000, color='black', linestyle='--', lw=1.5)
    plt.legend(loc="center right", markerscale=6., scatterpoints=1, fontsize=10)

    plt.tight_layout()
    plt.savefig("figs/scenario2_inst.pdf", bbox_inches='tight')


def ftl_instantaneous():
    """
    Plot the risk of one single run 
    """
    fname = "../checkpoints/scenario2/mlp_all_period20_errs.pkl"
    with open(fname, "rb") as fp:
        info = pickle.load(fp)

    errs = []
    for i in range(len(info)):
        tstep = info[i][0]
        errs.append(info[i][1][0, tstep])

    errs = np.array(errs)[0:200]
    times = np.arange(len(errs))

    for i in range(0, len(errs)+1, 10):
        # Vertical line
        plt.axvline(x=i, color='black', linestyle='--', lw=0.5)
        

    # vertical line at t=1000
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.85,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    plt.scatter(times, errs, label="Follow the leader", s=1, alpha=0.7)

    plt.ylabel("Instantaneous Risk")
    plt.xlabel("Time (t)")
    plt.legend(loc="center right", markerscale=7., scatterpoints=1, fontsize=15)

    plt.savefig("figs/scenario2_ftl_inst.pdf", bbox_inches='tight')


def finetune_ep20_instantaneous():
    """
    Plot the risk of one single run 
    """
    fname = "../checkpoints/scenario2/mlp_ft_p20_errs.pkl"
    with open(fname, "rb") as fp:
        info = pickle.load(fp)

    errs = []
    for i in range(len(info)):
        tstep = info[i][0]
        errs.append(info[i][1][0, tstep])


    errs = np.array(errs)[0:200]
    times = np.arange(len(errs))

    errs = np.cumsum(errs)
    errs = errs / np.arange(1, len(errs)+1)

    for i in range(0, len(errs)+1, 10):
        # Vertical line
        plt.axvline(x=i, color='black', linestyle='--', lw=0.5)
        

    # vertical line at t=1000
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.85,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    plt.scatter(times, errs, label="Fine-tuning", s=1, alpha=0.7)

    plt.ylabel("Instantaneous Risk")
    plt.xlabel("Time (t)")
    plt.legend(loc="center right", markerscale=7., scatterpoints=1, fontsize=15)

    plt.savefig("figs/scenario2_finetune_inst.pdf", bbox_inches='tight')


def finetune_ep1_instantaneous():
    """
    Plot the risk of one single run 
    """
    fname = "../checkpoints/scenario2/mlp_ft1_p20_errs.pkl"
    with open(fname, "rb") as fp:
        info = pickle.load(fp)

    errs = []
    for i in range(len(info)):
        tstep = info[i][0]
        errs.append(info[i][1][0, tstep])

    errs = np.array(errs)[0:200]
    times = np.arange(len(errs))

    # errs = np.cumsum(errs)
    # errs = errs / np.arange(1, len(errs)+1)

    for i in range(0, len(errs)+1, 10):
        # Vertical line
        plt.axvline(x=i, color='black', linestyle='--', lw=0.5)
        

    # vertical line at t=1000
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.85,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    plt.scatter(times, errs, label="Fine-tuning", s=1, alpha=0.7)

    plt.ylabel("Instantaneous Risk")
    plt.xlabel("Time (t)")
    plt.legend(loc="center right", markerscale=7., scatterpoints=1, fontsize=15)

    plt.savefig("figs/scenario2_finetune1_inst.pdf", bbox_inches='tight')


def online_learners():

    fnames = ["../checkpoints/scenario2/mlp_ft1_p20_errs.pkl",
              "../checkpoints/scenario2/mlp_all_period20_errs.pkl"]

    infos = []
    for fname in fnames:
        with open(fname, "rb") as fp:
            info = pickle.load(fp)
        infos.append(info)

    all_errs = []
    for info in infos:
        errs = []
        for i in range(len(info)):
            tstep = info[i][0]
            errs.append(info[i][1][0, tstep])
        all_errs.append(errs)


    all_errs[0] = np.cumsum(all_errs[0][0:200])
    all_errs[1] = np.cumsum(all_errs[1][0:200])

    all_errs[0] = all_errs[0] / np.arange(1, len(all_errs[0])+1)
    all_errs[1] = all_errs[1] / np.arange(1, len(all_errs[1])+1)
    all_errs = np.array(all_errs)

    times = np.arange(len(all_errs[0]))

    for i in range(0, len(all_errs[0])+1, 10):
        # Vertical line
        plt.axvline(x=i, color='black', linestyle='--', lw=0.5)
        
    # vertical line at t=1000
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.85,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    plt.plot(times, all_errs[0], c='C0')
    plt.plot(times, all_errs[1], c='C1')

    plt.scatter(times, all_errs[0], label="Follow-the-leader", s=1, alpha=0.7, c='C0')
    plt.scatter(times, all_errs[1], label="Online-SGD", s=1, alpha=0.7, c='C1')


    plt.ylabel("Risk")
    plt.xlabel("Time (t)")
    plt.legend(loc="upper right", markerscale=7., scatterpoints=1, fontsize=15)

    plt.savefig("figs/scenario2_online.pdf", bbox_inches='tight')

    plt.show()

# retro_vs_prospective()
# retro_vs_prospective_instantaneous()
# finetune_ep20_instantaneous()
# finetune_ep20_instantaneous()
# finetune_ep1_instantaneous()
online_learners()
