#1Dmodel.py
#
# Quick and dirty forward propagation of T and C in 1 dim

import numpy as np
import auxillary_functions as func
import constants as co


def T_next(T_prev: np.array, C_prev: np.array) -> np.array:
	return 1/(co.rho_m*co.cp_m) * (co.k_m*np.gradient(T_prev, co.dx)**2 - co.rho_w*co.cp_w*func.u_w(T_prev, C_prev)*np.gradient(T_prev, co.dx) )

def C_next(T_prev: np.array, C_prev: np.array) -> np.array:
	return co.D*np.gradient(C_prev, co.dx)**2 - np.gradient(C_prev * func.u_w(T_prev, C_prev), co.dx)


def G_S_2eq(n: int, f, g, dt, F0, G0):
	#Reserve memory for all calculations
	F, G = np.zeros(n), np.zeros(n)
	F[0], G[0] = F0, G0
	#Gauss-Seidel
	for i in range(n):
		F[i+1] = F[i] + dt*f(F[i], G[i])
		G[i+1] = G[i] + dt*g(F[i], G[i])
	return F, G

#How to run
T0 = np.zeros(20)
C0 = np.arange(20)
n = 100
T,C = G_S_2eq(n, T_next, C_next, co.dt, T0, C0)

print(T)
print(C)
