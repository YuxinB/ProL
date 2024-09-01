import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def make_plot(info, title, figname, size=50):
    pass
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.75,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})
    plt.figure(figsize=(5, 5))
    plt.ylim([-0.05, 1])
    plt.title(title)

    methods = []
    for m in info:
        methods.append(m)

    plt.ylabel("Prospective Risk")
    plt.xlabel("Time (t)")

    for i, m in enumerate(methods):
        plt.plot(info[m][2], info[m][0], c='C%d' % i)

    plt.axhline(y=0.0, color='black', linestyle='--')

    for i, m in enumerate(methods):
        plt.scatter(info[m][2], info[m][0], c='C%d' % i, s=size)
        std = 2 * info[m][1] / np.sqrt(5)
        mean = info[m][0]
        plt.fill_between(info[m][2], mean-std, mean+std,
                         alpha=0.3, color='C%d' % i)
    plt.legend(methods + ['Bayes risk'],
               loc="upper right", markerscale=2.,
               scatterpoints=1, fontsize=15, frameon=True)


    plt.savefig("./figs/aug20/%s.pdf" % figname, bbox_inches='tight')
    plt.show()

def synthetic_scenario2():
    info = np.load("./metrics/syn_scenario2.pkl", allow_pickle=True)
    make_plot(info, "Synthetic Scenario 2", figname="syn_scenario2")

def synthetic_scenario3():
    info = np.load("./metrics/syn_scenario3.pkl", allow_pickle=True)
    make_plot(info, "Synthetic Scenario 3", figname="syn_scenario3")

def mnist_scenario2():
    info = np.load("./metrics/mnist_scenario2.pkl", allow_pickle=True)
    make_plot(info, "MNIST Scenario 2", figname="mnist_scenario2")

def mnist_scenario3():
    info = np.load("./metrics/mnist_scenario3.pkl", allow_pickle=True)
    make_plot(info, "MNIST Scenario 3", figname="mnist_scenario3")

def cifar_scenario2():
    info = np.load("./metrics/cifar_scenario2.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 2", figname="cifar_scenario2")

def cifar_scenario3():
    info = np.load("./metrics/cifar_scenario3.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 3", figname="cifar_scenario3")

synthetic_scenario2()
synthetic_scenario3()
mnist_scenario2()
mnist_scenario3()
cifar_scenario2()
cifar_scenario3()
