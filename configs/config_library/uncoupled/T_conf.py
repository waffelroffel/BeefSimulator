# T_conf
# conv_oven

import numpy as np
import constants as c
from auxillary_functions import u_w
from auxillary_functions import f_func


def T_initial(T, C, shape, xx, yy, zz, t):
    return c.T_0


T_a = c.rho_m * c.cp_m

T_b = c.k_m

T_c = 0


def T_alpha(T, C, shape, xx, yy, zz, t):
    temp = c.k_m * np.ones(shape)
    # Symmetric B.C.
    temp[-1, :, :] = 1
    temp[:, -1, :] = 1
    return temp.ravel()


def T_beta(T, C, shape, xx, yy, zz, t):
    temp = np.zeros(shape)
    return temp.ravel()


def T_gamma(T, C, shape, xx, yy, zz, t):
    temp = np.ones(shape)
    # Bottom has a different heat transfer than the rest
    # temp[:,:,0] *= c.h_plate / c.h_air ## No; assume heat transfer is material independent
    temp[-1, :, :] = 0
    temp[:, -1, :] = 0
    return temp.ravel() * (1 - f_func(T)) * c.h * (c.T_oven - T)


def T_uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    return u.reshape((3, -1)).T


#T_bnd_types = ["n", "r", "n", "r", "r", "r"]
T_bnd_types = []

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
    # "analytic": T
}
