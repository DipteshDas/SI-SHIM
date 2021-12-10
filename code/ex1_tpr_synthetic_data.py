import numpy as np
from sklearn.model_selection import train_test_split
import stat_tools
import matplotlib.pyplot as plt
import pdb
import util_interactions
import lambda_path
import tau_path
from gen_synthetic_data import gen_design_matrix, gen_samples


def fs_in_A(fs, A, B):

    tmp1 = map(tuple, fs)
    tmp2 = map(tuple, A)
    tmp3 = map(tuple, B)

    tmp = list(set(tmp1).intersection(tmp2, tmp3))

    return [list(elm) for elm in tmp]


def plot(results, n_samples, filename):

    no_labels = [''] * len(n_samples)
    plt.figure(figsize=(8, 6))
    data = np.array(results)
    ds = data[:, 0, :]
    homo = data[:, 1, :]
    poly = data[:, 2, :]

    # pdb.set_trace()
    data_ds = [ds[i] for i in range(len(n_samples))]
    data_homo = [homo[i] for i in range(len(n_samples))]
    data_poly = [poly[i] for i in range(len(n_samples))]

    bp_ds = plt.boxplot(data_ds, patch_artist=True, positions=[3, 9, 15, 21], labels=no_labels,
                        boxprops=dict(facecolor="C0", alpha=0.5))
    bp_homo = plt.boxplot(data_homo, patch_artist=True, positions=[4, 10, 16, 22], labels=n_samples,
                          boxprops=dict(facecolor="C2", alpha=0.5))
    bp_poly = plt.boxplot(data_poly, patch_artist=True, positions=[5, 11, 17, 23], labels=no_labels,
                          boxprops=dict(facecolor="C3", alpha=0.5))

    plt.tick_params(bottom=False)
    ax = plt.gca()
    ax.legend([bp_ds["boxes"][0], bp_homo["boxes"][0], bp_poly["boxes"][0]], ['ds', 'homo', 'poly'], loc='lower right',
              prop={'size': 26})
    ax.set_xlim(0, 27)
    plt.ylabel("TPR", fontsize=30)
    plt.xlabel("Sample size", fontsize=30)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
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


def ds_inference(X_test, y_test, A, j_selected, sigma=1, sig_level=0.05, verbose=0):

    n, _ = X_test.shape
    cov = np.identity(n) * (sigma ** 2)

    XA = util_interactions.construct_XA(X_test, A)

    etaj, etajTy = util_interactions.construct_test_statistic(j_selected, XA, y_test, A)
    z_obs = etajTy
    n_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    if n_sigma == 0:
        return None

    p_value_naive = stat_tools.p_value(z_obs, 0, n_sigma, trunc_points=[[-np.inf, np.inf]])

    count = 0
    if p_value_naive < sig_level:
        count = 1

    # print("p_value_DS: {}, z_obs: {}, f: {} \n".format(round(p_value_naive, 2), z_obs, A[j_selected]))
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

    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))[0][0]

    if tn_sigma == 0:
        return None

    p_value_homo = stat_tools.p_value(z_obs, 0, tn_sigma, trunc_points=trunc_points_homo)
    count_homo = 0
    if p_value_homo < sig_level:
        count_homo = 1

    p_value_poly = stat_tools.p_value(z_obs, 0, tn_sigma, trunc_points=trunc_points_poly)
    count_poly = 0
    if p_value_poly < sig_level:
        count_poly = 1

    # print("p_value_homo: {}, p_value_poly: {}, z_obs: {}, f: {} \n".format(round(p_value_homo, 45),
    #                                                                        round(p_value_poly, 45), z_obs,
    #                                                                        A[j_selected]))
    return count_homo, count_poly


def tprs(X, y, A, fs, sigma=1, sig_level=0.05, lamda=1.0, alpha=0.0, max_depth=1, verbose=0):

    ds = ds_select(X, y, lamda, alpha, max_depth=max_depth)

    if ds is not None:
        X_test, y_test, A_ds = ds
    else:
        return None

    common_fs = fs_in_A(fs, A, A_ds)

    if len(common_fs) == 0:
        return None
    else:
        detect = 1

        indx = np.random.randint(len(common_fs))
        f = common_fs[indx]
        j_selected_ds = A_ds.index(f)
        count_ds = ds_inference(X_test, y_test, A_ds, j_selected_ds, sigma=sigma, sig_level=sig_level,
                                verbose=verbose)
        if count_ds is None:
            # print("count_ds is None !")
            return None

        j_selected_homo = A.index(f)
        homo_poly = homo_poly_inference(X, y, A, j_selected_homo, sigma, sig_level, lamda, alpha, max_depth,
                                        verbose=verbose)

        if homo_poly is None:
            # print("homo_poly is None !")
            return None
        else:
            count_homo, count_poly = homo_poly

        return [count_ds, count_homo, count_poly, detect]


def run(X, y, max_depth, lamda, alpha, sigma, sig_level, fs, verbose=0):

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

    tpr = tprs(X, y, A, fs, sigma=sigma, sig_level=sig_level, lamda=lamda, alpha=alpha, max_depth=max_depth,
               verbose=verbose)

    if tpr is None:
        # print("TPRs are None !!!")
        return None
    else:
        return tpr


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

        tpr_dss = []
        tpr_homos = []
        tpr_polys = []

        for i in range(5):
            dss, hms, pls, detect = 0, 0, 0, 0

            itr = 0
            while itr < max_iter:
                y = gen_samples(X, true_y, sigma, 1).flatten()
                res = run(X, y, max_depth, lamda, alpha, sigma, sig_level, fs, verbose=0)

                if res is not None:
                    print(itr)
                    itr += 1
                    [ds, hm, pl, dt] = res
                    # print(res)

                else:
                    continue

                dss += ds
                hms += hm
                pls += pl
                detect += dt

            tpr_ds = dss / detect
            tpr_homo = hms / detect
            tpr_poly = pls / detect

            tpr_dss.append(tpr_ds)
            tpr_homos.append(tpr_homo)
            tpr_polys.append(tpr_poly)

            print("tpr_ds: {}, tpr_homo: {}, tpr_poly: {}\n".format(tpr_ds, tpr_homo, tpr_poly))

        data = [tpr_dss, tpr_homos, tpr_polys]
        results.append(data)

    # pdb.set_trace()
    np.savez("../results/synthetic/results_tpr.npz", results=results)
    filename = "../results/synthetic/tpr.pdf"
    plot(results, n_samples, filename)
