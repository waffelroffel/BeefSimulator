# auxillary_functions.py
#
# Functions to consult for beef model

import numpy as np
import constants as co


def E(T: np.array) -> np.array:
    '''
    Calculate elasticity modulus
    :param T: np.array: Temperature distribution (x,y,z) in ⁰C
    :return: np.array: Elasticity modulus distribution (x,y,z) in Pa.
    '''
    return co.E0 + co.Em / (1 + np.exp(-co.En*(T-co.ED)))


def mu_w(T: np.array) -> np.array:
    '''
    Calculate water viscosity
    :param T: np.array: Temperature distribution (x,y,z) in ⁰C
    :return: np.array: Viscosity distribution (x,y,z)
    '''
    return np.exp(-0.0072*T-2.8658)


def C_eq(T: np.array) -> np.array:
    '''
    Calculate equilibrium water holding capacity
    :param T: np.array: Temperature distribution (x,y,z) in ⁰C
    :return: np.array: Equilibrium water holding capacity (x,y,z)
    '''
    return co.a1 - co.a2 / (1+co.a3*np.exp(-co.a4*(T-co.T_sig)))


def u_w(T: np.array, C: np.array, dh) -> np.array:
    '''
    Calculate water velocity in meat
    :param T: np.array: Temperature distribution (x,y,z) in ⁰C
    :param C: np.array: Water holding capacity distribution (x,y,z)
    :return: np.array: Water velocity vector [(x,y,z), (x,y,z), (x,y,z)]
    '''
    return -co.K * E(T) / mu_w(T) * np.array(np.gradient(C - C_eq(T), dh))


def div(A: np.array, dr: float = 1) -> np.array:
    '''
    Shorthand function for calculating divergence along first (0-th) axis
    :param A: np.array: N-dim - to be divergenced
    :param dr: float: Distance between discrete points in A
    :return: np.array: N-dim - has been divergenced
    '''
    return np.sum(np.gradient(A, dr), 0)


def dotND(A: np.array, B: np.array, axis=0):
    '''
    Shorthand function for dotting N-dim arrays along first (0-th) axis
    :param A: np.array: first array, mxN
    :param B: np.array: second array, mxN
    :return: np.array: N-dim where the m-dim axis 0 has been summed over
    '''
    return np.sum(A*B, axis)


def f_func(T: np.array) -> np.array:
    '''
    Calculate fraction of water used for evaporation
    :param T: np.array: Temperature distribution (x,y,z) in ⁰C
    :return: np.array: fraction of water used for evaporation (x,y,z)
    '''
    return co.f_max + (co.f0-co.f_max) / (1+np.exp((T-co.f1)/co.f2))


# def f_func(T: np.array) -> np.array:
#     f = np.zeros(T.shape)
#     f[T >= 100] = co.f
#     return f
