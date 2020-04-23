import numpy as np
from auxillary_functions import u_w

alp = 0.01  # 0.125e-6


def T(xx, yy, zz, t):
    return 3 * np.exp(-4*alp*(np.pi)**2*(1+1+4) * t) * \
        np.sin(2*np.pi * xx) * np.sin(2*np.pi * yy) * np.sin(4*np.pi * zz)


def T_initial(xx, yy, zz, t):
    return T(xx, yy, zz, t)


T_a = 1/alp

T_b = 1

T_c = 0


def T_alpha(xx, yy, zz, t):
    return 0


def T_beta(xx, yy, zz, t):
    return np.ones(xx.size)


def T_gamma(xx, yy, zz, t):
    return np.zeros(xx.size)


def T_uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    return u.reshape((3, -1)).T


T_bnd_types = ["d", "d", "d", "d", "d", "d"]

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
