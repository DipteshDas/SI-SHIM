import numpy as np
from numpy.linalg import inv, pinv
from queue import Queue
import time
import sys
import pdb
import util_interactions


def stepsize_inclusion(XA, Xt, y, beta, nu, alpha, A):

    tol = 1e-10
    W = y - XA.dot(beta)
    V = XA.dot(nu)
    Xk = XA[:, 0][:, np.newaxis]
    beta_k = beta[0]
    nu_k = nu[0]
    if ((Xt.T - Xk.T).dot(V) - alpha * nu_k) != 0:
        dt1 = ((Xt.T - Xk.T).dot(W) + alpha * beta_k) / ((Xt.T - Xk.T).dot(V) - alpha * nu_k)
        d1 = dt1.item()
        if 0 < d1 < tol:
            d1 = np.inf
    else:
        d1 = np.inf
    if ((Xt.T + Xk.T).dot(V) + alpha * nu_k) != 0:
        dt2 = ((Xt.T + Xk.T).dot(W) - alpha * beta_k) / ((Xt.T + Xk.T).dot(V) + alpha * nu_k)
        d2 = dt2.item()
        if 0 < d2 < tol:
            d2 = np.inf
    else:
        d2 = np.inf

    if d1 < 0 and d2 < 0:
        return np.inf
    elif d1 < 0:
        return d2
    elif d2 < 0:
        return d1
    else:
        return min(d1, d2)


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


def bound(XA, Xt, y, d_opt, beta, nu):
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

    max_bound = bw + d_opt * bv

    return max_bound


def check_bound_main_search(XA, Xt, y, d_opt, beta, nu, alpha):

    W = y - XA.dot(beta)
    V = XA.dot(nu)
    rho0 = XA[:, 0].T.dot(W) - alpha * beta[0]
    eta0 = XA[:, 0].T.dot(V) + alpha * nu[0]
    min_bound = np.abs(rho0) - d_opt * np.abs(eta0)
    max_bound = bound(XA, Xt, y, d_opt, beta, nu)

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


def explore(X, y, beta, nu, A, alpha=0.0, max_depth=None, search=0):
    """
    Construct the high-order interactions as a tree of patterns and then search it efficiently using branch and bound
    method.

    """

    p = X.shape[1]
    Q = Queue(maxsize=0)

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
                    gain = np.abs(Xt.T.dot(y))[0]
                    if gain > d_opt:
                        d_opt = gain
                        j_opt = j
                check_bound_flag = check_bound_initial_search(Xt, y, d_opt)

            else:
                if j not in A:
                    dt = stepsize_inclusion(XA, Xt, y, beta, nu, alpha, A)
                    # print("j: {}, dt = {}\n".format(j, dt))

                    if dt > 0 and dt < d_opt:
                        d_opt = dt
                        j_opt = j

                if d_opt == np.inf:  # If 'd_opt = np.inf', anyway 'check_bound_flag' will be 'True'
                    check_bound_flag = True
                else:
                    check_bound_flag = check_bound_main_search(
                        XA, Xt, y, d_opt, beta, nu, alpha)

            if check_bound_flag:
                child = node(level, pattern_child, depth + 1)
                Q.put_nowait(child)
                i += 1

    ts = time.process_time() - start_time  # Note time difference

    return d_opt, j_opt, i, ts


def initial_Search(X, y, beta, nu, A, max_depth=None):
    d_opt, j, counts, ts = explore(X, y, beta, nu, A, max_depth=max_depth, search=0)
    return d_opt, j, counts, ts


def main_Search(X, y, beta, nu, A, alpha=0.0, max_depth=None):
    d_opt, j, counts, ts = explore(X, y, beta, nu, A, alpha=alpha, max_depth=max_depth, search=1)
    return d_opt, j, counts, ts


def run_lambda_path(X, y, lmd_tgt=None, alpha=0.0, max_depth=None, verbose=0):
    """
    Construct the 'lambda-path' efficiently using 'homotopy-mining'.

    """

    beta = np.zeros(1)
    nu = np.zeros(1)

    node_counts = []
    tss = []
    node_traversed, ts_avg = 0, 0

    A = []
    lmd, j, counts, ts = initial_Search(X, y, beta, nu, A, max_depth=max_depth)
    node_counts.append(counts)
    tss.append(ts)

    if lmd < lmd_tgt:
        return None

    if j is not None:
        A.append(j)
    else:
        return None

    # Determine direction vector nu
    XA = util_interactions.construct_XA(X, A)
    Gram_aset = XA.T.dot(XA)
    cond = alpha * np.identity(len(A))
    iGram_aset = pinv(Gram_aset + cond)
    residual = y - XA.dot(beta)
    XTR = XA.T.dot(residual)
    eta = XTR / lmd
    nu = iGram_aset.dot(eta)

    # Update "lists"
    list_lmd = [lmd]
    list_beta = [beta.copy()]
    list_aset = [A.copy()]
    list_nu = [nu.copy()]

    if verbose >= 2:
        print("\n aset: {}\n".format(A))
        print(np.linalg.norm(X.T.dot(y), ord=np.inf), lmd)
        print("===============\n")

    while lmd > 0.:

        d2, j_out = stepsize_deletion(beta, nu)  # Step-size deletion
        d1, j_in, counts, ts = main_Search(X, y, beta, nu, A, alpha=alpha, max_depth=max_depth)  # Step-size inclusion

        node_counts.append(counts)
        tss.append(ts)
        # print("node counts (lamda-path) : {}, time: {}, pattern: {}\n".format(counts, ts, j_in))

        ts_day = 86400

        if np.sum(tss) >= ts_day:
            print("The 'lamda-path' did not finish in one day !!! \n")
            sys.exit()

        if d1 == np.Inf and d2 == np.Inf:  # Terminal case !
            d = lmd - lmd_tgt  # d is always positive ("lmd" is decreasing)
            beta = beta + d * nu  # Update 'beta'
            list_lmd.append(lmd_tgt)
            list_beta.append(beta.copy())
            list_aset.append(A.copy())  # "A" does not change from last kink
            list_nu.append(nu.copy())  # "nu" does not change from last kink

            break

        d = min(d1, d2)
        tol = 1e-11

        if d < tol:  # Exact regularization path breaks here !
            print("Debug!")
            pdb.set_trace()
        else:
            beta = beta + d * nu  # Update 'beta'
            lmd = lmd - d  # Update 'lmd'

        # Return output for specific "lambda" value
        if (lmd_tgt is not None) and (lmd <= lmd_tgt):
            if lmd < lmd_tgt:
                beta = list_beta[-1].copy()
                lmd = list_lmd[-1].copy()

                d = d - 1e-10
                beta = beta + d * nu  # Update 'beta'
                lmd = lmd - d  # Update 'lmd'

                delta = (lmd_tgt - list_lmd[-1]) / (lmd - list_lmd[-1])
                beta_tgt = delta * beta + (1 - delta) * list_beta[-1]
            else:
                beta_tgt = beta

            # residual = y - X.dot(beta_tgt)  # Update the error
            XA = util_interactions.construct_XA(X, A)
            residual = y - XA.dot(beta_tgt)  # Update the error
            XTR = XA.T.dot(residual) - alpha * beta_tgt
            idx_active = np.where(np.isclose(np.abs(XTR), lmd_tgt))[0].tolist()
            if len(idx_active) == 0:
                return None
            A = [A[i] for i in idx_active]

            # Compute direction vector "nu"
            XA = util_interactions.construct_XA(X, A)
            Gram_aset = XA.T.dot(XA)
            cond = alpha * np.identity(len(A))
            iGram_aset = pinv(Gram_aset + cond)
            XTR = XA.T.dot(residual) - alpha * beta_tgt
            eta = XTR / lmd_tgt
            nu = iGram_aset.dot(eta)

            list_lmd.append(lmd_tgt)
            list_beta.append(beta_tgt.copy())
            list_aset.append(A.copy())
            list_nu.append(nu.copy())

            if verbose == -3:
                node_traversed = np.mean(node_counts)
                ts_avg = np.mean(tss)

            return list_lmd, list_beta, list_aset, list_nu, node_traversed, ts_avg

        if d == d1:  # Inclusion
            A.append(j_in)  # Update active set
            beta = np.append(beta, np.zeros(1))
            if verbose >= 2:
                print("Pattern included: {}, d1 = {}, d2 = {} and beta: {}, nu: {}".format(
                    j_in, d1, d2, beta, nu))

        else:  # Deletion
            if verbose >= 2:
                print("Pattern deleted: {}, d1 = {}, d2 = {} and beta: {}, nu: {}".format(j_out, d1, d2, beta, nu))

            del A[j_out]  # Update A
            beta = np.delete(beta, j_out)  # Update beta

        if len(A) != 0:
            XA = util_interactions.construct_XA(X, A)
            residual = y - XA.dot(beta)  # Update the error

            # Compute direction vector "nu"
            Gram_aset = XA.T.dot(XA)
            cond = alpha * np.identity(len(A))
            iGram_aset = pinv(Gram_aset + cond)
            XTR = XA.T.dot(residual) - alpha * beta
            eta = XTR / lmd
            nu = iGram_aset.dot(eta)
            # pdb.set_trace()

            # Update "lists"
            list_lmd.append(lmd.copy())
            list_beta.append(beta.copy())
            list_aset.append(A.copy())
            list_nu.append(nu.copy())

        else:
            print("Debug: A is empty !")
            pdb.set_trace()

    if verbose == -3:
        node_traversed = np.mean(node_counts)
        ts_avg = np.mean(tss)

    return list_lmd, list_beta, list_aset, list_nu, node_traversed, ts_avg
