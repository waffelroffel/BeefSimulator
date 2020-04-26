import numpy as np
from auxillary_functions import u_w

D = 4e-10
Lx = 0.075
Ly = 0.039
Lz = 0.054
# D = diffusion coeff. This produces the eq. dC/dt = D*grad^2(D)


def C(xx, yy, zz, t):
    return 3 * np.exp(-4*D*(np.pi)**2*(1/Lx+1/Ly+4/Lz) * t) * \
        np.sin(2*np.pi/Lx * xx) * np.sin(2*np.pi/Ly * yy) * np.sin(4*np.pi/Lz * zz)


def C_initial(T, C, shape, xx, yy, zz, t):
    # return 3 * np.sin(2*np.pi/Lx * xx) * np.sin(2*np.pi/Ly * yy) * np.sin(4*np.pi/Lz * zz)
    return 0


C_a = 1

C_b = 1

C_c = 0


def C_alpha(T, C, shape, xx, yy, zz, t):
    return 0


def C_beta(T, C, shape, xx, yy, zz, t):
    return 1


def C_gamma(T, C, shape, xx, yy, zz, t):
    return 0


def C_uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    return u.reshape((3, -1)).T


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
    "analytic": C
}
