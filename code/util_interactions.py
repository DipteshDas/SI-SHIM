import numpy as np
from itertools import combinations


def construct_test_statistic(j, XA, y, A):
    ej = np.zeros(len(A))[:, np.newaxis]
    ej[j] = 1

    inv = np.linalg.pinv(np.dot(XA.T, XA))
    XAinv = np.dot(XA, inv)
    etaj = np.dot(XAinv, ej)

    y = y[:, np.newaxis]
    etajTy = np.dot(etaj.T, y)[0][0]

    return etaj, etajTy


def compute_yz(y, etaj, zk, sigma):
    n = y.shape[0]
    y = y[:, np.newaxis]
    cov = np.identity(n) * sigma ** 2
    denom = np.dot(np.dot(etaj.T, cov), etaj)
    tau_hat = cov.dot(etaj) / denom
    eps_0 = np.dot(np.identity(n) - tau_hat.dot(etaj.T), y)
    yz = eps_0 + tau_hat * zk

    return yz.flatten(), tau_hat.flatten()


def construct_interval(list_zk, list_active_set, A):
    z_interval = []
    tmp1 = [tuple(item) for item in A]
    for i in range(len(list_zk) - 1):
        tmp2 = [tuple(item) for item in list_active_set[i]]
        if set(tmp1) == set(tmp2):
            z_interval.append([list_zk[i], list_zk[i + 1] - 1e-5])

    new_z_interval = []
    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    return new_z_interval


def construct_interval_polytope(list_zk, list_active_set, A, z_obs):
    z_interval = []
    tmp1 = [tuple(item) for item in A]
    for i in range(len(list_zk) - 1):
        tmp2 = [tuple(item) for item in list_active_set[i]]
        if list_zk[i] <= z_obs <= list_zk[i + 1]:
            if set(tmp1) == set(tmp2):
                z_interval.append([list_zk[i], list_zk[i + 1] - 1e-5])

    new_z_interval = []
    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    return new_z_interval


def construct_XA(X, A):
    j = A[0]
    XA = np.prod(X[:, j], 1)[:, np.newaxis]

    i = 1
    while i < len(A):
        j = A[i]
        XA = np.column_stack((XA, np.prod(X[:, j], 1)[:, np.newaxis]))
        i += 1

    return XA
