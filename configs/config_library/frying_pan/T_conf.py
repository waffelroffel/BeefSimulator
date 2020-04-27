# T_conf
# frying_pan

import numpy as np
import constants as c
from auxillary_functions import u_w
from auxillary_functions import f_func


def T_initial(T, C, shape, xx, yy, zz, t):
    return c.T_0


T_a = c.rho_m * c.cp_m

T_b = c.k_m

T_c = - c.rho_w * c.cp_w


def T_alpha(T, C, shape, xx, yy, zz, t):
    temp = c.k_m * np.ones(shape)
    # Symmetric B.C.
    temp[-1, :, :] = 1
    temp[:, -1, :] = 1
    return temp.flatten()


def T_beta(T, C, shape, xx, yy, zz, t):
    temp = c.cp_w * c.rho_w * np.ones(shape)
    # Symmetric B.C.
    temp[-1, :, :] = 0
    temp[:, -1, :] = 0
    return temp.flatten()


def T_gamma(T, C, shape, xx, yy, zz, t):
    temp = np.ones(shape)
    # Symmetric B.C.
    temp[-1, :, :] = 0
    temp[:, -1, :] = 0
    T_shape = T.reshape(shape)

    # Air facing edges compare to room temperature
    temp[:, :, 1:] *= (1 - f_func(T_shape[:, :, 1:])) * \
        c.h * (c.T_room - T_shape[:, :, 1:])

    # Bottom facing edge compare to pan temperature
    temp[:, :, 0] = (1-f_func(T_shape[:, :, 0])) * \
        c.h * (c.T_pan - T_shape[:, :, 0])
    # TODO: Check if h should be different for the pan (or sous vide)
    return temp.flatten()


def T_uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    return u.reshape((3, -1)).T


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
}
