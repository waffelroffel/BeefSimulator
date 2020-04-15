## C_conf

import numpy as np
from auxillary_functions import u_w, C_eq


def C(xx, yy, zz, t):
    return 1


def C_initial(xx, yy, zz, t):
    return 0.75 # kg water/kg beef
np.ones(xx.size)

def C_a(xx, yy, zz, t):
    return np.ones(xx.size)


def C_b(xx, yy, zz, t):
    return - u_w * np.ones(xx.size)


def C_c(xx, yy, zz, t):
    return c.D * np.ones(xx.size)


def C_alpha(xx, yy, zz, t, TT, CC):
    return - c.D * np.ones(xx.size)


def C_beta(xx, yy, zz, t, TT, CC):
    temp = ( u_w - c.f * c.h_air * (c.T_oven - TT)/(c.H_evp * c.rho_w) )
    temp[:,:,0] = ( u_w - c.f * c.h_plate * (c.T_oven - TT[:,:,0])/(c.H_evp * c.rho_w) )
    return temp.flatten()


def C_gamma(xx, yy, zz, t):
    temp = - c.f * c.h_air * (c.T_oven - TT)/(c.H_evp * c.rho_w) * C_eq(TT)
    temp[:,:,0] *= c.h_plate / c.h_air
    return temp.flatten()


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
