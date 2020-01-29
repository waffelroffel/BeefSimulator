#1Dmodel.py
#
# Quick and dirty forward propagation of T and C in 1 dim

import numpy as np
import auxillary_functions as func

k_m = 1
rho_w = 1
cp_w = 1
delta_t = 0.1
D = 1
rho_m = 1
cp_m = 1
dx = 0.1


def T_next(T_prev: np.array, C_prev: np.array) -> np.array:
	return 1/(rho_m*cp_m) * (k_m*np.gradient(T_prev, dx)**2 - rho_w*cp_w*func.u_w(T_prev, C_prev)*np.gradient(T_prev, dx) )

def C_next(T_prev: np.array, C_prev: np.array) -> np.array:
	return D*np.gradient(C_prev, dx)**2 - np.gradient(C_prev * func.u_w(T_prev, C_prev), dx)


def G_S_2eq(n: int, f, g, dt, F0, G0):
	#Reserve memory for all calculations
	F, G = np.zeros(n), np.zeros(n)
	F[0], G[0] = F0, G0
	#Gauss-Seidel
	for i in range(n):
		F[i+1] = F[i] + dt*f(F[i], G[i])
		G[i+1] = G[i] + dt*g(F[i], G[i])
	return F, G

T0 = np.zeros(20)
C0 = np.arange(20)
T,C = G_S_2eq(100, T_next, C_next, delta_t, T0, C0)

print(T)
print(C)