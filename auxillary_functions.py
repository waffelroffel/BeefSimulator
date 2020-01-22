#auxillary_functions.py
#
#Functions to consult for beef model

import numpy as np


# Elasticity modulus
def E(T: np.array) -> np.array:
	'''
	
	:param T: np.array: Temperature distribution (x,y,z) in ⁰C
	:return: np.array: Elasticity modulus distribution (x,y,z) in Pa.
	'''
	E0 = 12e3 #Pa
	Em = 83e3 #Pa
	En = 0.3
	ED = 60
	
	return E0 + Em / (1 + np.exp(-En*(T-ED)))


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
	a1 = 0.745
	a2 = 0.345
	a3 = 30
	a4 = 0.25
	T_sig = 52 #⁰C
	
	return a1 - a2 / (1+a3*np.exp(-a4*(T-T_sig)))


# Fluid velocity
def u_w(T: np.array, C: np.array) -> np.array:
	K = 1e-17 #Permeability [m²] - in range 1e-17 to 1e-19
	return -K * E(T) / mu_w(T) * ..... #TODO: finish implementation
