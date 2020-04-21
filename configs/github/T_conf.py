# T_conf
# conv_oven

import numpy as np
import constants as c
from auxillary_functions import u_w
from auxillary_functions import f_func

mu = 0.01


def T_analytic(T, C, shape, xx, yy, zz, t):
    return 3 * np.exp(-4*mu*(np.pi)**2*(1+1+4) * t) * np.cos(2*np.pi * xx) * np.cos(2*np.pi * yy) * np.cos(4*np.pi * zz) + 20


def T_initial(T, C, shape, xx, yy, zz, t):
    return T_analytic(T, C, shape, xx, yy, zz, t)


def T_a(T, C, shape, xx, yy, zz, t):
    return 1/mu


def T_b(T, C, shape, xx, yy, zz, t):
    return 1


def T_c(T, C, shape, xx, yy, zz, t):
    return 0


def T_alpha(T, C, shape, xx, yy, zz, t):
    return 1


def T_beta(T, C, shape, xx, yy, zz, t):
    return 0


def T_gamma(T, C, shape, xx, yy, zz, t):
    return 0


def T_uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    return np.zeros_like(u.reshape((3, -1)).T)


T_bnd_types = ["n", "n", "n", "n", "n", "n"]

T_conf = {
    "pde": {"a": T_a,
            "b": T_b,
            "c": T_c},
    "bnd": {"alpha": T_alpha,
            "beta": T_beta,
            "gamma": T_gamma},
    "uw": T_uw,
    "bnd_types": T_bnd_types,
    "initial": T_initial,
    "analytic": T_analytic
}
