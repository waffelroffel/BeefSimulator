#1Dmodel.py
#
# Quick and dirty forward propagation of T and C in 1 dim

import numpy as np
import auxillary_functions as func

k_m = 1
rho_w = 1
cp_w = 1
dt = 0.1
D = 1

def T_next(T_prev: np.array, C_prev: np.array) -> np.array:
	return T_prev + dt * (k_m*np.gradient(T_prev)**2 - rho_w*cp_w*func.u_w(T_prev, C_prev)*np.gradient(T_prev) )

def C_next(T_prev: np.array, C_prev: np.array) -> np.array:
	return C_prev + dt * (D*np.gradient(C_prev)**2 - np.gradient(C_prev * func.u_w(T_prev, C_prev)))
