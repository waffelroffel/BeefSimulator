import numpy as np


def T(xx, yy, zz, t):
    return 1000 * 3 * np.exp(-4*(np.pi)**2*(1+1+4) * t) * \
        np.sin(2*np.pi * xx) * np.sin(2*np.pi * yy) * np.sin(4*np.pi * zz)


def T_initial(xx, yy, zz, t):
    return T(xx, yy, zz, t)


def T_a(xx, yy, zz, t):
    return np.ones(xx.size)


def T_b(xx, yy, zz, t):
    return np.ones(xx.size)


def T_c(xx, yy, zz, t):
    return np.zeros(xx.size)


def T_alpha(xx, yy, zz, t):
    return np.zeros(xx.size)


def T_beta(xx, yy, zz, t):
    return np.ones(xx.size)


def T_gamma(xx, yy, zz, t):
    return np.zeros(xx.size)


def T_ux(xx, yy, zz, t):
    return 1


def T_uy(xx, yy, zz, t):
    return 1


def T_uz(xx, yy, zz, t):
    return 1


T_bnd_types = ["d", "d", "d", "d", "d", "d"]

T_conf = {
    "pde": {"a": T_a,
            "b": T_b,
            "c": T_c},
    "bnd": {"alpha": T_alpha,
            "beta": T_beta,
            "gamma": T_gamma},
    "u": {"x": T_ux,
          "y": T_uy,
          "z": T_uz},
    "bnd_types": T_bnd_types,
    "initial": T_initial,
    "analytic": T
}
