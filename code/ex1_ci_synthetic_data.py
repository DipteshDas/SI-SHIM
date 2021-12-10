import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pdb
import stat_tools
import util_interactions
import lambda_path
import tau_path
from gen_synthetic_data import gen_design_matrix, gen_samples


def plot(data, fig_path):

    plt.figure(figsize=(8, 6))
    labels = ['ds', 'homo', 'poly']
    colors = ['C0', 'C2', 'C3']

    bp = plt.boxplot(data, patch_artist=True, labels=labels)

    for patch, median, color in zip(bp['boxes'], bp['medians'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        median.set(color='r', lw=1)

    ax = plt.gca()

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylabel("CI length", fontsize=30)
    plt.xlabel("Methods", fontsize=30)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_ylim(0, 6)

    plt.tight_layout()

    plt.savefig(fig_path, dpi=300)
    plt.show()


def fs_in_A(fs, A):

    tmp1 = map(tuple, fs)
    tmp2 = map(tuple, A)
    tmp3 = list(set(tmp1).intersection(tmp2))

    return tmp3


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


def ds_inference(X_test, y_test, A, j_selected, sigma, sig_level=0.05, verbose=0):

    n, _ = X_test.shape
    cov = np.identity(n) * (sigma ** 2)

    XA = util_interactions.construct_XA(X_test, A)

    etaj, etajTy = util_interactions.construct_test_statistic(j_selected, XA, y_test, A)
    z_obs = etajTy
    if z_obs == 0:
        return [None, None]

    n_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]
    L_n, U_n = stat_tools.ci(z_obs, n_sigma, [[-np.inf, np.inf]], sig_level, verbose=verbose)

    return L_n, U_n


def ci_all(X, y, A, sig_level, sigma, lamda, alpha, max_depth, verbose=0):

    ds = ds_select(X, y, lamda, alpha, max_depth=max_depth, verbose=verbose)

    if ds is not None:
        X_test, y_test, A_ds = ds
    else:
        return None

    tmp = fs_in_A(A, A_ds)

    if len(tmp) == 0:
        # print("No common selected feature found !")
        return None

    for indx in range(len(tmp)):
        f = tmp[indx]

        j_selected_homo = A.index(list(f))
        j_selected_ds = A_ds.index(list(f))

        n, _ = X.shape
        cov = np.identity(n) * (sigma ** 2)

        XA = util_interactions.construct_XA(X, A)
        etaj, etajTy = util_interactions.construct_test_statistic(j_selected_homo, XA, y, A)

        model = call_tau_path(X, y, etaj, etajTy, sigma, lamda=lamda, alpha=alpha,
                              max_depth=max_depth, verbose=verbose)

        if model is None:
            return None

        list_zk, list_active_set_new = model

        z_obs = etajTy

        trunc_points_homo = util_interactions.construct_interval(list_zk, list_active_set_new, A)
        trunc_points_poly = util_interactions.construct_interval_polytope(list_zk, list_active_set_new, A, z_obs)

        tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

        L_h, U_h = stat_tools.ci(z_obs, tn_sigma, trunc_points_homo, sig_level, verbose=0)
        if None not in [L_h, U_h]:
            cil_homo = U_h - L_h
        else:
            continue

        L_p, U_p = stat_tools.ci(z_obs, tn_sigma, trunc_points_poly, sig_level, verbose=0)
        if None not in [L_p, U_p]:
            cil_poly = U_p - L_p
        else:
            continue

        L_ds, U_ds = ds_inference(X_test, y_test, A_ds, j_selected_ds, sigma, sig_level=sig_level, verbose=0)
        if None not in [L_ds, U_ds]:
            cil_ds = U_ds - L_ds
            return [cil_homo, cil_poly, cil_ds]
        else:
            return None


def run(X, y, max_depth, lamda, alpha, sigma, sig_level, verbose=0):

    model = lambda_path.run_lambda_path(X, y, lmd_tgt=lamda, alpha=alpha,
                                        max_depth=max_depth, verbose=verbose)

    if model is None:
        return None

    list_lmd, list_beta, list_aset, _, _, _ = model

    A = list_aset[-1].copy()

    if len(A) == 0:
        # print("length of A is zero")
        return None

    cis = ci_all(X, y, A, sig_level, sigma, lamda=lamda, alpha=alpha, max_depth=max_depth, verbose=verbose)

    if cis is None:
        # print("Confidence interval is None !!!")
        return None
    else:
        return cis


def exp_params():
    n = 100
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

    return n, r, max_depth, fs, beta_vec, proba, zeta, sigma


if __name__ == '__main__':

    lamda = 1
    alpha = 0.01 * lamda
    sig_level = 0.05

    max_iter = 100

    cil_polys = []
    cil_homos = []
    cil_dss = []

    n, r, max_depth, fs, beta_vec, proba, zeta, sigma = exp_params()
    X, true_y = gen_design_matrix(n, r, fs, beta_vec, proba, zeta=zeta, random_state=42)  # Fixed 'X' & 'mu'

    itr = 0
    while itr < max_iter:
        y = gen_samples(X, true_y, sigma, 1).flatten()
        res = run(X, y, max_depth, lamda, alpha, sigma, sig_level, verbose=0)

        if res is not None:
            print(itr)
            itr += 1
            [cil_homo, cil_poly, cil_ds] = res
            print(res)
        else:
            # print("None \n")
            continue

        cil_homos += [cil_homo]
        cil_polys += [cil_poly]
        cil_dss += [cil_ds]

    cil_mean_ds, cil_std_ds = np.mean(cil_dss), np.std(cil_dss)
    cil_mean_homo, cil_std_homo = np.mean(cil_homos), np.std(cil_homos)
    cil_mean_poly, cil_std_poly = np.mean(cil_polys), np.std(cil_polys)

    print("\n\n cil_ds: {} ({}), cil_homo: {} ({}), cil_poly: {} ({})\n".format(cil_mean_ds,
                                                                                cil_std_ds,
                                                                                cil_mean_homo,
                                                                                cil_std_homo,
                                                                                cil_mean_poly,
                                                                                cil_std_poly))

    data = [cil_dss, cil_homos, cil_polys]
    res_path = "../results/synthetic/"
    file_data = res_path + "data_ci.npz"
    fig_path = res_path + "ci.pdf"
    np.savez(file_data, results=data)
    plot(data, fig_path)
