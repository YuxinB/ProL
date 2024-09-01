import numpy as np


def get_bayes(p):
    """
    Calculate bayes risk on markov chain
    """

    Tr = np.array([[p, 1-p], [1-p, p]])

    pi = np.array([1, 0])

    err = []
    for i in range(10):
        err.append(np.min(pi))
        pi = pi @ Tr

    print(p, np.mean(err))


get_bayes(0.1)
get_bayes(0.2)



