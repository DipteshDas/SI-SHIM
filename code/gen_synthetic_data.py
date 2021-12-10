import numpy as np


def gen_design_matrix(n, p, fs, beta, proba, zeta=0.0, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    indx_range = np.arange(0, p)

    X, true_y = [], []

    for i in range(n):
        x_i = [0] * p

        if proba is not None:
            indx = np.random.choice(indx_range, size=int(p * (1 - zeta)), p=proba)
        else:
            indx = np.random.choice(indx_range, size=int(p * (1 - zeta)))

        for j in indx:
            x_i[j] = 1

        mu = 0
        for j in range(len(beta)):
            mu += beta[j] * np.prod(np.array(x_i)[fs[j]])

        X.append(x_i)
        true_y.append(mu)

    return np.array(X), np.array(true_y)


def gen_samples(X, mu, sigma, max_iter):

    n, _ = X.shape
    y = np.zeros((n, max_iter))

    for i in range(n):
        y[i, :] = np.random.normal(mu[i], sigma, size=max_iter)

    return y
