#auxillary_functions.py
#
#Functions to consult for beef model

import numpy as np
import constants as co


# Elasticity modulus
def E(T: np.array) -> np.array:
	'''
	
	:param T: np.array: Temperature distribution (x,y,z) in ⁰C
	:return: np.array: Elasticity modulus distribution (x,y,z) in Pa.
	'''
	return co.E0 + co.Em / (1 + np.exp(-co.En*(T-co.ED)))


# Viscosity of water
def mu_w(T: np.array) -> np.array:
	'''
	
	:param T: np.array: Temperature distribution (x,y,z) in ⁰C
	:return: np.array: Viscosity distribution (x,y,z)
	'''
	return np.exp(-0.0072*T-2.8658)


# Equilibrium water holding capacity
def C_eq(T: np.array) -> np.array:
	'''
	
	:param T: np.array: Temperature distribution (x,y,z) in ⁰C
	:return: np.array: Equilibrium water holding capacity (x,y,z)
	'''
	return co.a1 - co.a2 / (1+co.a3*np.exp(-co.a4*(T-co.T_sig)))

# Fluid velocity
def u_w(T: np.array, C: np.array) -> np.array:
	return -co.K * E(T) / mu_w(T) * np.gradient(C - C_eq(T), co.dx)

