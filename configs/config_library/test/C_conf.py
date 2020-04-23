import numpy as np
from auxillary_functions import u_w


def C(xx, yy, zz, t):
    return 1


def C_initial(xx, yy, zz, t):
    return C(xx, yy, zz, t)


C_a = 1

C_b = 1

C_c = 0


def C_alpha(xx, yy, zz, t):
    return np.zeros(xx.size)


def C_beta(xx, yy, zz, t):
    return np.ones(xx.size)


def C_gamma(xx, yy, zz, t):
    return np.zeros(xx.size)


def C_uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    return u.reshape((3, -1)).T


C_bnd_types = []


C_conf = {
    "pde": {"a": C_a,
            "b": C_b,
            "c": C_c},
    "bnd": {"alpha": C_alpha,
            "beta": C_beta,
            "gamma": C_gamma},
    "uw": C_uw,
    "bnd_types": C_bnd_types,
    "initial": C_initial,
    # "analytic": C
}
