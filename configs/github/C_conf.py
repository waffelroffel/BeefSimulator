# C_conf
# conv_oven

import numpy as np
import constants as c
from auxillary_functions import u_w
from auxillary_functions import f_func

mu = 0.01


def C_analytic(T, C, shape, xx, yy, zz, t):
    return 3 * np.exp(-4*mu*(np.pi)**2*(1+1+4) * t) * np.sin(2*np.pi * xx) * np.sin(2*np.pi * yy) * np.sin(4*np.pi * zz) + 20


def C_initial(T, C, shape, xx, yy, zz, t):
    return C_analytic(T, C, shape, xx, yy, zz, t)


C_a = 1/mu

C_b = 1

C_c = 0


def C_alpha(T, C, shape, xx, yy, zz, t):
    return 0


def C_beta(T, C, shape, xx, yy, zz, t):
    return 1


def C_gamma(T, C, shape, xx, yy, zz, t):
    return 20


def C_uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    return np.zeros_like(u.reshape((3, -1)).T)


C_bnd_types = ["d", "d", "d", "d", "d", "d"]

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
    "analytic": C_analytic
}
