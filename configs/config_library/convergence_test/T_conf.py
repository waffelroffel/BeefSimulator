import numpy as np
from auxillary_functions import u_w

alp = 0.125e-6
# All sidelengths are 1
# alp = alpha = heat transfer coeff. This produces the eq. dT/dt = alpha*grad^2(T)


def T(xx, yy, zz, t):
    return 3 * np.exp(-4*alp*(np.pi)**2*(1+1+4) * t) * \
        np.sin(2*np.pi * xx) * np.sin(2*np.pi * yy) * np.sin(4*np.pi * zz)


def T_initial(T, C, shape, xx, yy, zz, t):
    return T(xx, yy, zz, t)


T_a = 1

T_b = alp

T_c = 0


def T_alpha(T, C, shape, xx, yy, zz, t):
    return 0


def T_beta(T, C, shape, xx, yy, zz, t):
    return 1


def T_gamma(T, C, shape, xx, yy, zz, t):
    return 0


def uw(T, C, I, J, K, dh):
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
    "uw": uw,
    "bnd_types": T_bnd_types,
    "initial": T_initial,
    "analytic": T
}
