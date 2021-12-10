import numpy as np
from numpy.linalg import inv, pinv
from queue import Queue
import time
import pdb
import util_interactions
from lambda_path import run_lambda_path


def stepsize_inclusion(XA, Xt, y, beta, nu, lmd, b):

    tol = 1e-10
    W = y - XA.dot(beta)  # error
    s_j = np.dot(Xt.T, W) / lmd
    gamma_j = np.dot(Xt.T, b - np.dot(XA, nu))
    if gamma_j != 0:
        d = (lmd * (np.sign(gamma_j) - s_j)) / gamma_j
        d = d.item()
        if 0 < d < tol:
            d = np.inf
    else:
        d = np.inf
    # print("d = {}, s_j = {}, gamma_j = {} \n".format(d, s_j, gamma_j))
    return d


def stepsize_deletion(beta, nu):

    tol = 1e-10
    indx_nz = np.where(beta != 0)[0].tolist()  # Search over only active features
    beta_nz = beta[indx_nz]
    nu_nz = nu[indx_nz]
    step_sizes = []
    for i in range(len(indx_nz)):
        if nu_nz[i] != 0:
            tmp = - beta_nz[i] / nu_nz[i]
            if tmp > 0:  # "step-size" must be positive !
                step_sizes.append(tmp)
            else:
                step_sizes.append(np.inf)
        else:
            step_sizes.append(np.inf)
    if len(step_sizes) > 0:
        indx_min = np.argmin(step_sizes)
        d2 = step_sizes[indx_min]
        j_out = indx_nz[indx_min]
        if 0 < d2 < tol:
            d2 = np.inf
            j_out = None
    else:
        d2 = np.inf
        j_out = None

    return d2, j_out


def bound(XA, Xt, y, d_opt, beta, nu, b):
    """
    Compute the upper bound used in the tree pruning.

    """

    W = y - XA.dot(beta)
    indx_pos_W = np.where(W > 0)[0]
    indx_neg_W = np.where(W < 0)[0]

    bw_pos = Xt[indx_pos_W].T.dot(np.abs(W[indx_pos_W]))
    bw_neg = Xt[indx_neg_W].T.dot(np.abs(W[indx_neg_W]))
    bw = max(bw_pos, bw_neg)

    V = XA.dot(nu)
    indx_pos_V = np.where(V > 0)[0]
    indx_neg_V = np.where(V < 0)[0]

    bv_pos = Xt[indx_pos_V].T.dot(np.abs(V[indx_pos_V]))
    bv_neg = Xt[indx_neg_V].T.dot(np.abs(V[indx_neg_V]))
    bv = max(bv_pos, bv_neg)

    indx_pos_b = np.where(b > 0)[0]
    indx_neg_b = np.where(b < 0)[0]

    theta_pos = Xt[indx_pos_b].T.dot(np.abs(b[indx_pos_b]))
    theta_neg = Xt[indx_neg_b].T.dot(np.abs(b[indx_neg_b]))
    bth = max(theta_pos, theta_neg)

    max_bound = bw + d_opt * bth + d_opt * bv

    return max_bound


def check_bound_main_search(XA, Xt, y, d_opt, beta, nu, alpha, j, b):
    W = y - XA.dot(beta)
    V = XA.dot(nu)
    rho0 = XA[:, 0].T.dot(W) - alpha * beta[0]
    eta0 = XA[:, 0].T.dot(V) + alpha * nu[0]
    min_bound = np.abs(rho0) - d_opt * np.abs(XA[:, 0].T.dot(b)) - d_opt * np.abs(eta0)
    max_bound = bound(XA, Xt, y, d_opt, beta, nu, b)

    check_bound_flag = False
    if max_bound > min_bound:  # i.e. if feasible solution exists
        check_bound_flag = True

    return check_bound_flag


def check_bound_initial_search(Xt, y, max_corr):
    indx_pos = np.where(y > 0)[0]
    indx_neg = np.where(y < 0)[0]

    # Find upper bound of corr
    corr_pos = Xt[indx_pos].T.dot(np.abs(y[indx_pos]))
    corr_neg = Xt[indx_neg].T.dot(np.abs(y[indx_neg]))

    check_bound_flag = False
    if max(corr_pos, corr_neg) > max_corr:
        check_bound_flag = True

    return check_bound_flag


class node(object):
    """
    Class representing each node of the tree used in 'explore'.

    """

    def __init__(self, level, pattern, depth):
        self.level = level
        self.pattern = pattern
        self.depth = depth


def explore(X, y, beta, nu, A, lmd, b, alpha, max_depth=None, search=0, verbose=0):
    """
    Construct the high-order interactions as a tree of patterns and then search it efficiently using branch and bound
    method.

    """

    # pdb.set_trace()
    p = X.shape[1]
    Q = Queue(maxsize=0)
    # pdb.set_trace()
    if max_depth is None:
        max_depth = p
    else:
        max_depth = max_depth
    root_pattern = [0] * p

    if search == 0:
        d_opt = 0
        j_opt = None
    else:
        XA = util_interactions.construct_XA(X, A)
        d_opt = np.Inf
        j_opt = None
    root = node(-1, root_pattern, 0)
    Q.put(root)
    i = 0

    start_time = time.process_time()  # Note start time
    while not Q.empty():
        parent = Q.get_nowait()
        level = parent.level
        pattern = parent.pattern
        depth = parent.depth

        while level < p - 1 and depth < max_depth:
            level += 1
            pattern_child = pattern[:level] + [1] + pattern[level + 1:]
            j = np.where(np.array(pattern_child) == 1)[0].tolist()
            Xt = np.prod(X[:, j], 1)[:, np.newaxis]

            if np.sum(Xt) == 0:  # If a whole column is zero, then we can safely ignore 'j' and it's descendants.
                continue

            if search == 0:
                if j not in A:
                    corr = np.abs(Xt.T.dot(y))[0]
                    if corr > d_opt:
                        d_opt = corr
                        j_opt = j
                check_bound_flag = check_bound_initial_search(Xt, y, d_opt)
            else:
                if j not in A:
                    dt = stepsize_inclusion(XA, Xt, y, beta, nu, lmd, b)
                    if dt > 0 and dt < d_opt:
                        d_opt = dt
                        j_opt = j

                if d_opt == np.inf:  # If 'd_opt = np.inf', anyway 'check_bound_flag' will be 'True'
                    check_bound_flag = True
                else:
                    check_bound_flag = check_bound_main_search(
                        XA, Xt, y, d_opt, beta, nu, alpha, j, b)

            if check_bound_flag:
                child = node(level, pattern_child, depth + 1)
                Q.put_nowait(child)
                if verbose >= 2:
                    print("pattern : {},  i = {}".format(pattern_child, i))
                i += 1

    ts = time.process_time() - start_time  # Note time difference
    return d_opt, j_opt, i, ts


def run_tau_path(X, y, etaj, sigma, lmd, alpha=0.0, max_depth=None, z_min=-20, z_max=20, verbose=0):
    """
    Construct the 'tau-path' efficiently using 'homotopy-mining'.

    """

    zk = z_min  # start from "z_min" and then increase gradually to "z_max"

    node_counts, tss = [], []
    node_traversed, ts_avg = 0, 0

    # Initial search
    yz, b = util_interactions.compute_yz(y, etaj, zk, sigma)
    model = run_lambda_path(X, yz, lmd_tgt=lmd, alpha=alpha, max_depth=max_depth, verbose=verbose)

    if model is None:
        return None

    list_lmd_init, list_beta_init, list_aset_init, _, _, _ = model

    if len(list_aset_init) == 0:
        # print("Length of A: {}\n".format(0))
        return None

    A = list_aset_init[-1]
    beta = list_beta_init[-1]
    XA = util_interactions.construct_XA(X, A)

    # Compute direction vector "nu"
    cond = alpha * np.identity(len(A))
    C_inv = pinv(XA.T.dot(XA) + cond)
    invXAzT = C_inv.dot(XA.T)
    nu = invXAzT.dot(b)

    # Update lists (Initial update)
    list_zk = [zk]
    list_aset = [A.copy()]
    list_beta = [beta.copy()]
    list_nu = [nu.copy()]

    while zk < z_max:
        d2, j_out = stepsize_deletion(beta, nu)  # Step-size deletion
        d1, j_in, counts, ts = explore(X, yz, beta, nu, A, lmd, b, alpha, max_depth, search=1)  # Step-size inclusion

        node_counts.append(counts)
        tss.append(ts)

        if d1 <= 0:
            d1 = np.inf
            j_in = None
        if d2 <= 0:
            d2 = np.inf
            j_out = None

        if d1 == np.Inf and d2 == np.Inf:

            if zk != z_max:
                zk = z_max  # capped at "z_max"
                yz, b = util_interactions.compute_yz(y, etaj, zk, sigma)  # Update "yz"
                model = run_lambda_path(X, yz, lmd_tgt=lmd, alpha=alpha, max_depth=max_depth, verbose=verbose)

                if model is None:
                    print("Debug! \n")
                    pdb.set_trace()

                lmd_list, beta_list, aset_list, _, _, _ = model
                A = aset_list[-1]
                beta = beta_list[-1]

                # "nu" does not change, but the order of selected features in A returned by "lambda-path" might be
                # different. Hence, we need to recompute 'nu'.
                XA = util_interactions.construct_XA(X, A)

                # Compute direction vector "nu"
                cond = alpha * np.identity(len(A))
                C_inv = pinv(XA.T.dot(XA) + cond)
                invXAzT = C_inv.dot(XA.T)
                nu = invXAzT.dot(b)

                list_zk.append(zk)
                list_aset.append(A.copy())
                list_beta.append(beta.copy())
                list_nu.append(nu.copy())

            break

        d = min(d1, d2)

        tol = 1e-11
        if d < tol:
            return None
        else:
            # UPDATE (beta, zk, yz)
            beta = beta + d * nu  # Update 'beta'
            zk = zk + d  # Update "zk"
            yz, b = util_interactions.compute_yz(y, etaj, zk, sigma)  # Update "yz"

        if d == d1:  # Inclusion
            A.append(j_in)  # Update active set
            beta = np.append(beta, np.zeros(1))
            if verbose >= 2:
                print("Pattern Included: {}, zk = {}, A = {},  d1 = {}, d2 = {}".format(j_in, zk, A, d1, d2))
        else:  # Deletion
            pattern_del = A[j_out]
            del A[j_out]  # Update A
            beta = np.delete(beta, j_out)  # Update beta
            if verbose >= 2:
                print("Pattern Deleted: {}, zk = {}, A = {},  d1 = {}, d2 = {}".format(pattern_del, zk, A, d1, d2))

        if len(A) != 0:
            XA = util_interactions.construct_XA(X, A)

            # Compute direction vector "nu"
            cond = alpha * np.identity(len(A))
            C_inv = pinv(XA.T.dot(XA) + cond)
            invXAzT = C_inv.dot(XA.T)
            nu = invXAzT.dot(b)

            # Update all lists
            if zk <= z_max:
                list_zk.append(zk)
                list_aset.append(A.copy())
                list_beta.append(beta.copy())
                list_nu.append(nu.copy())
            else:
                zk = z_max  # capped at "z_max"
                yz, b = util_interactions.compute_yz(y, etaj, zk, sigma)  # Update "yz"

                model = run_lambda_path(X, yz, lmd_tgt=lmd, alpha=alpha, max_depth=max_depth, verbose=verbose)

                if model is None:
                    print("Debug! \n")
                    pdb.set_trace()

                lmd_list, beta_list, aset_list, _, _, _ = model

                A = aset_list[-1].copy()
                beta = beta_list[-1].copy()
                # nu = list_nu[-1].copy()  # "nu" does not change at this point. That's the whole point!

                # "nu" does not change, but the order of selected features in A returned by "lambda-path" might be
                # different. Hence, we need to recompute 'nu'.
                XA = util_interactions.construct_XA(X, A)

                # Compute direction vector "nu"
                cond = alpha * np.identity(len(A))
                C_inv = pinv(XA.T.dot(XA) + cond)
                invXAzT = C_inv.dot(XA.T)
                nu = invXAzT.dot(b)

                list_zk.append(zk)
                list_aset.append(A.copy())
                list_beta.append(beta.copy())
                list_nu.append(nu)  # "nu" does not change at this point. That's the whole point!

        else:
            # print("A is empty !")
            if verbose >= 2:
                print("A is empty !")
            repeat = 0
            while zk < z_max:
                # pdb.set_trace()
                repeat += 1
                zk = zk + repeat * 1e-1
                yz, b = util_interactions.compute_yz(y, etaj, zk, sigma)  # Update "yz"
                model = run_lambda_path(X, yz, lmd_tgt=lmd, alpha=alpha, max_depth=max_depth, verbose=verbose)

                if model is None:
                    continue

                lmd_list, beta_list, aset_list, _, _, _ = model

                if len(aset_list) != 0:
                    A = aset_list[-1].copy()
                    beta = beta_list[-1].copy()

                    XA = util_interactions.construct_XA(X, A)

                    # Compute direction vector "nu"
                    cond = alpha * np.identity(len(A))
                    C_inv = pinv(XA.T.dot(XA) + cond)
                    invXAzT = C_inv.dot(XA.T)
                    nu = invXAzT.dot(b)

                    # Update lists (Reinitialization)
                    list_zk.append(zk)
                    list_aset.append(A.copy())
                    list_beta.append(beta.copy())
                    list_nu.append(nu)  # "nu" does not change at this point. That's the whole point!
                    break

            else:
                break

    if verbose == -2:
        node_traversed = np.mean(node_counts)
        ts_avg = np.mean(tss)

    return list_zk, list_beta, list_aset, list_nu, node_traversed, ts_avg
