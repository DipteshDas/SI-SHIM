import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pdb
import util_interactions
import lambda_path
import tau_path
import stat_tools
from gen_synthetic_data import gen_design_matrix, gen_samples


def exp_params():
    r = 30
    max_depth = 3
    zeta = 0.95
    sigma = 1

    fs = [[0], [1, 2], [3, 4, 5]]  # true model
    beta_vec, exp = np.array([0.0, 0.0, 0.0]).tolist(), "fpr"  # true beta

    proba_val = {0: 0.1, 1: 0.15, 2: 0.15, 3: 0.2, 4: 0.2, 5: 0.2}
    proba = np.zeros(r)
    for i in proba_val.keys():
        proba[i] = proba_val[i]

    return r, max_depth, fs, beta_vec, proba, zeta, sigma


def plot_line(results, n_samples):
    plt.figure(figsize=(8, 6))
    x = np.arange(len(n_samples))

    data = np.array(results)
    ds = data[:, 0, :]
    homo = data[:, 1, :]
    poly = data[:, 2, :]

    data_ds = [np.mean(ds[i]) for i in range(len(n_samples))]
    data_homo = [np.mean(homo[i]) for i in range(len(n_samples))]
    data_poly = [np.mean(poly[i]) for i in range(len(n_samples))]

    plt.plot(x, data_ds, marker='o', markerfacecolor='C0', markersize=6, color='C0', alpha=1, linewidth=2,
             label='ds')
    plt.plot(x, data_homo, marker='o', markerfacecolor='C2', markersize=6, color='C2', alpha=1, linewidth=2,
             label='homo')
    plt.plot(x, data_poly, marker='o', markerfacecolor='C3', markersize=6, color='C3', alpha=1, linewidth=2,
             label='poly')

    plt.xticks(x, n_samples)
    ax = plt.gca()
    ax.set_ylim(0, 0.3)
    plt.legend(fontsize=26)
    plt.ylabel("FPR", fontsize=30)
    plt.xlabel("Sample size", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.savefig("../results/synthetic/fpr.pdf", dpi=300)
    plt.show()


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


def ds_select(X, y, lamda, alpha, max_depth=1, verbose=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = lambda_path.run_lambda_path(X_train, y_train, lmd_tgt=lamda, alpha=alpha,
                                        max_depth=max_depth, verbose=verbose)

    if model is None:
        return None

    list_lmd, list_beta, list_aset, _, _, _ = model

    A = list_aset[-1].copy()

    if len(A) == 0:
        # print("length of A is zero in Data Splitting...")
        return None

    return X_test, y_test, A


def ds_inference(X_test, y_test, A, j_selected, sigma=1, sig_level=0.05):
    n, _ = X_test.shape
    cov = np.identity(n) * (sigma ** 2)

    XA = util_interactions.construct_XA(X_test, A)

    etaj, etajTy = util_interactions.construct_test_statistic(j_selected, XA, y_test, A)
    z_obs = etajTy
    n_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    if np.sum(etaj) == 0:
        return None

    p_value_naive = stat_tools.p_value(z_obs, 0, n_sigma, trunc_points=[[-np.inf, np.inf]])

    count = 0
    if p_value_naive < sig_level:
        count = 1

    return count


def homo_poly_inference(X, y, A, j_selected, sigma, sig_level, lamda, alpha, max_depth, verbose=0):
    n, _ = X.shape
    cov = np.identity(n) * (sigma ** 2)

    XA = util_interactions.construct_XA(X, A)

    etaj, etajTy = util_interactions.construct_test_statistic(j_selected, XA, y, A)

    model = call_tau_path(X, y, etaj, etajTy, sigma=1, lamda=lamda, alpha=alpha, max_depth=max_depth, verbose=verbose)

    if model is None:
        return None

    list_zk, list_active_set_new = model

    z_obs = etajTy

    trunc_points_homo = util_interactions.construct_interval(list_zk, list_active_set_new, A)
    trunc_points_poly = util_interactions.construct_interval_polytope(list_zk, list_active_set_new, A, z_obs)

    if len(trunc_points_poly) == 0:
        return None

    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    p_value_homo = stat_tools.p_value(z_obs, 0, tn_sigma, trunc_points=trunc_points_homo)
    count_homo = 0
    if p_value_homo < sig_level:
        count_homo = 1

    p_value_poly = stat_tools.p_value(z_obs, 0, tn_sigma, trunc_points=trunc_points_poly)
    count_poly = 0
    if p_value_poly < sig_level:
        count_poly = 1

    # print("p_value_homo: {}, p_value_poly: {}, z_obs: {}, f: {} \n".format(round(p_value_homo, 2),
    #                                                                 round(p_value_poly, 2), z_obs, A[j_selected]))

    return count_homo, count_poly


def fprs(X, y, A, sigma=1, sig_level=0.05, lamda=1.0, alpha=0.0, max_depth=1, verbose=0):
    ds = ds_select(X, y, lamda, alpha, max_depth=max_depth, verbose=verbose)

    if ds is not None:
        X_test, y_test, A_ds = ds
    else:
        return None

    indx = np.random.randint(len(A))
    f = A[indx]
    j_selected_homo = A.index(f)

    indx = np.random.randint(len(A_ds))
    f = A_ds[indx]
    j_selected_ds = A_ds.index(f)

    count_ds = ds_inference(X_test, y_test, A_ds, j_selected_ds, sigma=sigma, sig_level=sig_level)

    homo_poly = homo_poly_inference(X, y, A, j_selected_homo, sigma, sig_level, lamda, alpha, max_depth,
                                    verbose=verbose)

    if homo_poly is not None:
        count_homo, count_poly = homo_poly
    else:
        return None

    return [count_ds, count_homo, count_poly]


def run(X, y, max_depth, lamda, alpha, sigma, sig_level, verbose=0):
    n, _ = X.shape

    model = lambda_path.run_lambda_path(X, y, lmd_tgt=lamda, alpha=alpha,
                                        max_depth=max_depth, verbose=verbose)

    if model is None:
        return None

    list_lmd, list_beta, list_aset, _, _, _ = model
    A = list_aset[-1].copy()

    if len(A) == 0:
        # print("length of A is zero")
        return None

    fpr = fprs(X, y, A, sigma=sigma, sig_level=sig_level, lamda=lamda, alpha=alpha, max_depth=max_depth,
               verbose=verbose)

    if fpr is None:
        return None
    else:
        return fpr


if __name__ == '__main__':

    lamda = 1
    alpha = 0.01 * lamda
    sig_level = 0.05

    max_iter = 100

    n_samples = [100, 200, 400, 500]

    results = []

    r, max_depth, fs, beta_vec, proba, zeta, sigma = exp_params()

    for n in n_samples:
        print("=========== n: {} =============\n".format(n))
        X, true_y = gen_design_matrix(n, r, fs, beta_vec, proba, zeta=zeta, random_state=42)  # Fixed 'X' & 'mu'

        fpr_ds = []
        fpr_hm = []
        fpr_pl = []

        for i in range(5):

            dss, hms, pls = 0, 0, 0

            itr = 0

            while itr < max_iter:

                y = gen_samples(X, true_y, sigma, 1).flatten()
                res = run(X, y, max_depth, lamda, alpha, sigma, sig_level, verbose=0)

                if res is not None:
                    print(itr)
                    itr += 1
                    [ds, hm, pl] = res
                    # print(res)
                else:
                    continue

                dss += ds
                hms += hm
                pls += pl

            print("ds: {}, homo: {}, poly: {}\n".format(dss, hms, pls))

            fpr_ds += [dss / max_iter]
            fpr_hm += [hms / max_iter]
            fpr_pl += [pls / max_iter]

        print("ds: {} ({}), homo: {} ({}), poly: {} ({})\n".format(np.mean(fpr_ds), np.std(fpr_ds),
                                                                   np.mean(fpr_hm), np.std(fpr_hm),
                                                                   np.mean(fpr_pl), np.std(fpr_pl)))

        data = [fpr_ds, fpr_hm, fpr_pl]
        results.append(data)

    np.savez("../results/synthetic/results_fpr.npz", results=results)
    plot_line(results, n_samples)
