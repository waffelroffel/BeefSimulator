# C_conf
# frying_pan

import numpy as np
from auxillary_functions import u_w, C_eq
import constants as c
from auxillary_functions import f_func


def C_initial(T, C, shape, xx, yy, zz, t):
    return c.C_0


C_a = 1

C_b = c.D

C_c = -1


def C_alpha(T, C, shape, xx, yy, zz, t):
    temp = - c.D * np.ones(shape)
    # No flux through bottom
    temp[:, :, 0] = 1
    # Symmetric B.C.
    temp[-1, :, :] = 1
    temp[:, -1, :] = 1
    return temp


def C_beta(T, C, shape, xx, yy, zz, t):
    temp = np.ones(shape)
    # No flux through bottom
    temp[:, :, 0] = 0
    # Symmetric B.C.
    temp[-1, :, :] = 0
    temp[:, -1, :] = 0
    return temp


def C_gamma(T, C, shape, xx, yy, zz, t):
    temp = - np.ones(shape)
    # No flux through bottom
    temp[:, :, 0] = 0
    # Symmetric B.C.
    temp[-1, :, :] = 0
    temp[:, -1, :] = 0
    return temp.flatten() * f_func(T) * c.h * (c.T_room - T)/(c.H_evap * c.rho_w) * (C - C_eq(T))


def C_uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    return u.reshape((3, -1)).T


# C_bnd_types = ['n', 'n', 'n', 'n', 'd', 'n']
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
