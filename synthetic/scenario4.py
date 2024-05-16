import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.auto as tqdm

"""
We must make a sequence of predictions using (Z<t, a<t)

Similar to scenario 3, X_t = 1 (constant)
and Y_t = 0 or 1

There is MDP such that
- Transition matrix:  T(Y_{t-1}, h_t) -> Y_t
- Loss             :  L(Y_t, h_t) -> |Y_t - h_t|

The goal is a to find a sequence h_{>t} that minimizes the expected discounted loss
"""



class Data_Scenario4():
    def __init__(self, p=0.9, τ=30, max_t=1000):
        self.p = [[p, 1-p], [1-p, p]]
        self.τ = τ
        self.max_t = max_t
        gamma = 0.9
        self.gamma_vals = np.array([gamma**i for i in range(max_t)])


class MDP():
    def __init__(self):
        α = 0.7
        β = 0.1

        # T(a_t, s_t) = P(s_{t+1} | s_t, a_t)
        T = np.array([
            [[α, 1-α], [1-α, α]],
            [[β, 1-β], [1-β, β]],
        ])
        self.state = 0

    def step(self, state, action):
        y_prev = self.state

        y_next = np.random.choice([0, 1], p=T[action, y_prev])
        reward = np.abs(action - y_next)
        self.state = y_next

        return y_next, loss



def get_data(t):
    Z_t = []
    A_t = []
    L_t = []
    for i in range(t):
        action = np.random.choice([0, 1])
        z_true, loss = mdp.step(mdp.state, action)
        A_t.append(action)
        Z_t.append(z_true)
        L_t.append(loss)
    return Z_t, A_t, L_t


def erm_mdp(Z_t, ntest):
    p_hat = np.mean(samples)
    y_hat = p_hat > 0.5
    return np.ones(ntest) * y_hat


def prospective_mdp(Z_t, A_t, ntest):

    def solve_MDP(T, R, γ=0.9):
        # Q(a, s)
        Q_vals = np.zeros((2, 2))
        for iter in range(500):
            Q_vals_new = np.zeros_like(Q_vals)
            for s in range(2):
                for a in range(2):
                    for s_prime in range(2):
                        Q_new = T[a, s, s_prime] * (R[a, s_prime] + γ * np.min(Q_vals, axis=0))
                    Q_vals_new[a, s] = Q_new

            if np.max(np.abs(Q_vals - Q_vals_new)) < 1e-5:
                break
            Q_vals = Q_vals_new

        return Q_vals

    #T[a, s, s']
    T_hat = np.zeros((2, 2, 2))

    # L[a, s']
    L_hat = np.zeros((2, 2))

    for i in range(len(Z_t)-1):
        a_t = A_t[i]
        z_t = Z_t[i]
        z_t1 = Z_t[i+1]

        T_hat[a_t, z_t, z_t1] += 1
        L_hat[a_t, z_t1] += L_t[i]

    Q_star = solve_MDP(T_hat, L_hat)

    preds = []
    z_dist = np.array([[1 - z_t, z_t]])

    for s in range(ntest):

        # Find current action for distribution over states z_dst
        h_s = np.argmin(z_dist @ Q_star)
        preds.append(h_s)

        # update distribution over state
        z_dist = T[h_s] @ z_dist

    return preds







def eval_mdp(pred,  z_t):
    mdp = MDP()
    gamma = 0.9
    mdp.state = z_t

    total_loss = 0
    for i in range(len(pred)):
        _, loss = mdp.step(mdp.state, pred[i]))
        total_loss = loss + gamma * total_loss

    return total_loss



def main():
    mdp = MDP()

    seeds = 100
    max_t = 500
    run_t = 40
    times = np.arange(1, run_t-1)

    for t in range(2, run_t)
    for t in tqdm.tqdm(range(1, run_t)):

        for it in range(seeds):
            Z_t, A_t, L_t = get_data(it)
            ntest = max_t - len(Z_t)

            erm_pred = erm_mdp(Z_t, ntest)

            pr_pred = prospective_mdp(Z_t, A_t, ntest)



        for t in times:
            pass






if __name__ == "__main__":
    main()
