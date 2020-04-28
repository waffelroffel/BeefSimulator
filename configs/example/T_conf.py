import numpy as np
from auxillary_functions import u_w

mu = 0.01


def T_analytic(T, C, shape, xx, yy, zz, t):
    return 3 * np.exp(-4*mu*(np.pi)**2*(1+1+4) * t) * np.cos(2*np.pi * xx) * np.cos(2*np.pi * yy) * np.cos(4*np.pi * zz) + 20


def T_initial(T, C, shape, xx, yy, zz, t):
    return T_analytic(T, C, shape, xx, yy, zz, t)


T_a = 1/mu

T_b = 1

T_c = 0


def T_alpha(T, C, shape, xx, yy, zz, t):
    temp = np.ones(shape)
    temp[:, -1, :] = 0
    return temp


def T_beta(T, C, shape, xx, yy, zz, t):
    temp = np.zeros(shape)
    temp[:, -1, :] = 1
    return temp


def T_gamma(T, C, shape, xx, yy, zz, t):
    temp = np.zeros(shape)
    temp[:, -1, :] = 20
    return temp


# only needs to be defined one time.
# either in T_conf or C_conf
# T_conf takes priority
def T_uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    # temporary way to decouple the equation
    return np.ones(u.reshape((3, -1)).T.shape)
    # return u.reshape((3, -1)).T


# [x0, "y0", "z0", "xn", "yn", "zn"]
T_bnd_types = ["n", "n", "n", "n", "d", "n"]

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
