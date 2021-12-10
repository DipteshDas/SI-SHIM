import numpy as np
import lambda_path
import tau_path
import util_interactions
import matplotlib.pyplot as plt
import statsmodels.api as sm
import time
import stat_tools
from gen_synthetic_data import gen_design_matrix, gen_samples
import pdb


def call_tau_path(X, y, etaj, etajTy, sigma=1, lamda=1, alpha=0.0, max_depth=1, verbose=0):
    z_obs = etajTy
    [z_min, z_max] = [-np.abs(z_obs) - 20 * sigma, np.abs(z_obs) + 20 * sigma]

    model = tau_path.run_tau_path(X, y, etaj, sigma, lamda, alpha=alpha, max_depth=max_depth,
                                  z_min=z_min, z_max=z_max, verbose=verbose)

    if model is None:
        return None

    list_zk, beta_zks, list_aset_zk, list_nu, _, _ = model

    _, p = X.shape
    list_active_set_new = []
    for i in range(len(list_zk)):
        A_zk_list = list_aset_zk[i]
        list_active_set_new.append(A_zk_list)

    return list_zk, list_active_set_new


def run(X, y, true_y):
    lamda = 1
    alpha = 0.01 * lamda
    sigma = 1
    cov = np.identity(n)

    model = lambda_path.run_lambda_path(X, y, lmd_tgt=lamda, alpha=alpha, max_depth=max_depth)

    if model is None:
        return None

    list_lmd, list_beta, list_aset, _, _, _ = model

    A = list_aset[-1]

    if len(A) == 0:
        print("\n A is empty! \n")
        return None

    j_selected = np.random.randint(len(A))

    XA = util_interactions.construct_XA(X, A)

    etaj, etajTy = util_interactions.construct_test_statistic(j_selected, XA, y, A)
    model = call_tau_path(X, y, etaj, etajTy, sigma, lamda=lamda, alpha=alpha, max_depth=max_depth, verbose=0)

    if model is None:
        return None

    list_zk, list_active_set_new = model
    trunc_points_homo = util_interactions.construct_interval(list_zk, list_active_set_new, A)

    true_y = true_y.reshape((n, 1))
    tn_mu = np.dot(etaj.T, true_y)[0][0]
    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]
    pivot = stat_tools.cdf(etajTy, tn_mu, tn_sigma, trunc_points_homo)

    return pivot


def exp_params():
    r = 30
    max_depth = 3
    zeta = 0.95
    sigma = 1

    fs = [[0], [1, 2], [3, 4, 5]]  # true model
    beta_vec, exp = np.array([0.5, - 2.0, 3.0]).tolist(), "tpr"  # true beta

    proba_val = {0: 0.1, 1: 0.15, 2: 0.15, 3: 0.2, 4: 0.2, 5: 0.2}
    proba = np.zeros(r)
    for i in proba_val.keys():
        proba[i] = proba_val[i]

    return r, max_depth, fs, beta_vec, proba, zeta, sigma


if __name__ == '__main__':
    max_iter = 1000
    list_pivot = []

    r, max_depth, fs, beta_vec, proba, zeta, sigma = exp_params()
    n = 100
    X, true_y = gen_design_matrix(n, r, fs, beta_vec, proba, zeta=zeta, random_state=42)  # Fixed 'X' & 'mu'
    Y = gen_samples(X, true_y, sigma, max_iter)  # Generate 'max_iter' random 'Y' samples

    star_time = time.time()
    for each_iter in range(max_iter):
        print(each_iter)
        y = Y[:, each_iter]
        pivot = run(X, y, true_y)
        if pivot is not None:
            list_pivot.append(pivot)

    print("length of list_pivot: {}\n".format(len(list_pivot)))

    plt.rcParams.update({'font.size': 18})
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, sm.distributions.ECDF(np.array(list_pivot))(grid), 'r-', linewidth=5, label='Pivot ElNet TN-A')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    print("Time taken = {}\n".format(time.time() - star_time))
    # plt.savefig('../results/synthetic/pivot_LASSO_TN_A_3rd_order.png', dpi=100)
    plt.savefig('../results/synthetic/pivot_ElNet_TN_A_3rd_order.png', dpi=100)
    plt.show()
