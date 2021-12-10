import numpy as np
from mpmath import mp
import pdb

mp.dps = 1000


def get_area(a, b, loc, scale):

    Fa = mp.ncdf(a, loc, scale)
    Fb = mp.ncdf(b, loc, scale)

    return Fb - Fa


def cdf(z_obs, x, sigma, trunc_points=[[-np.inf, np.inf]]):
    """
    Compute a Gaussian cumulative distribution function truncated at a union
    of disjoint intervals.

    trunc_points must be a list of list
    """

    num, denum = 0, 0

    for i in np.arange(len(trunc_points)):
        a, b = trunc_points[i]
        prob_ab = get_area(a, b, loc=x, scale=sigma)

        if a <= z_obs < b:
            F_ab = get_area(a, z_obs, loc=x, scale=sigma)
        elif z_obs < a:
            F_ab = 0
        else:
            F_ab = prob_ab

        num += F_ab
        denum += prob_ab

    if denum != 0:
        f = num / denum
    else:
        f = 0

    return f


def find_root(z_obs, sigma, z_interval, y, lb, ub, n_itr, tol=1e-6, verbose=0):
    """
    Find the root corresponding to the upper or lower bound of confidence interval.

    """

    a, b = lb, ub
    diff = ub - lb

    if len(z_interval) == 1:
        if abs(z_obs - z_interval[0][0]) < 1e-4 or abs(z_obs - z_interval[0][1]) < 1e-4:
            return None

    fa, fb = cdf(z_obs, a, sigma, trunc_points=z_interval), cdf(z_obs, b, sigma, trunc_points=z_interval)
    if verbose:
        print("Start: fa: {}, fb: {}, a: {:.3f}, b: {:.3f}, z_obs: {}, z_interval: {}\n".
              format(round(fa, 3), round(fb, 3), a, b, z_obs, z_interval))

    itr = 0
    if (fa > y) and (fb > y):
        while fb > y:
            if abs(z_obs - b) < tol or itr >= n_itr:
                return None
            b = b + diff
            fb = cdf(z_obs, b, sigma, trunc_points=z_interval)
            if verbose:
                print("fa: {}, fb: {}, a: {:.3f}, b: {:.3f}, z_obs: {}, z_interval: {}\n".
                      format(round(fa, 3), round(fb, 3), a, b, z_obs, z_interval))
            itr += 1

    elif (fa < y) and (fb < y):

        while fa < y:

            if abs(z_obs - a) < tol or abs(fa - fb) < tol or itr >= n_itr:
                return None

            if fa < y and fa > fb:
                b = a
                fb = fa
                a = a - diff
            else:
                diff = (b - a) / 2 - tol
                a = a + diff

            fa = cdf(z_obs, a, sigma, trunc_points=z_interval)

            if verbose:
                print("fa: {}, fb: {}, a: {:.3f}, b: {:.3f}, z_obs: {}, z_interval: {}\n".
                      format(round(fa, 3), round(fb, 3), a, b, z_obs, z_interval))
            itr += 1

    if verbose:
        print("outside: fa: {}, fb: {}, a: {:.3f}, b: {:.3f}, z_obs: {}\n".format(round(fa, 3), round(fb, 3), a, b,
                                                                                  z_obs))

    max_iter = int(np.ceil((np.log(tol) - np.log(b - a)) / np.log(0.5)))

    c = None
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = cdf(z_obs, c, sigma, trunc_points=z_interval)
        # print("fc: {}, y: {:.3f}, a: {:.3f}, b: {:.3f}, c:{:.3f}\n".format(round(fc, 3), y, a, b, c))
        if fc > y:
            a = c
        elif fc < y:
            b = c

    return c


def ci(z_obs, sigma, trunc_points=[[-np.inf, np.inf]], alpha=0.05, n_itr=100, verbose=0):
    """
    Compute a (1 - alpha) confidence interval for a Gaussian distribution
    truncated at a union of disjoint intervals

    trunc_points must be a list of list
    """

    lb = z_obs - 20. * sigma
    ub = z_obs + 20. * sigma

    L = find_root(z_obs, sigma, trunc_points, 1.0 - 0.5 * alpha, lb, ub, n_itr, verbose=verbose)
    U = find_root(z_obs, sigma, trunc_points, 0.5 * alpha, lb, ub, n_itr, verbose=verbose)

    return np.array([L, U])


def p_value(z_obs, mu, sigma, trunc_points=[[-np.inf, np.inf]]):
    """
    Compute p-value for a Gaussian distribution truncated at a union of disjoint intervals

    trunc_points must be a list of list
    """

    value = cdf(z_obs, mu, sigma, trunc_points)
    return 2 * min(1 - value, value)


def ci_len(z_obs, sigma, trunc_points, sig_level):

    ci_int = ci(z_obs, sigma, trunc_points, sig_level)

    if None not in ci_int:
        L, U = ci_int
        return U - L
    else:
        return np.inf
