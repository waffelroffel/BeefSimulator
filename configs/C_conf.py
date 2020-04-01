import numpy as np


def C(xx, yy, zz, t):
    return 1


def C_initial(xx, yy, zz, t):
    return C(xx, yy, zz, t)


def C_a(xx, yy, zz, t):
    return np.ones(xx.size)


def C_b(xx, yy, zz, t):
    return np.ones(xx.size)


def C_c(xx, yy, zz, t):
    return np.zeros(xx.size)


def C_alpha(xx, yy, zz, t):
    return np.zeros(xx.size)


def C_beta(xx, yy, zz, t):
    return np.ones(xx.size)


def C_gamma(xx, yy, zz, t):
    return np.zeros(xx.size)


def C_ux():
    ...


def C_uy():
    ...


def C_uz():
    ...


C_bnd_types = []


C_conf = {
    "pde": {"a": C_a,
            "b": C_b,
            "c": C_c},
    "bnd": {"alpha": C_alpha,
            "beta": C_beta,
            "gamma": C_gamma},
    "u": {"x": C_ux,
          "y": C_uy,
          "z": C_uz},
    "bnd_types": C_bnd_types,
    "initial": C_initial,
    "analytic": C
}
