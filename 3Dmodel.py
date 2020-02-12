#3Dmodel.py
#
#Quick and dirty way to solve the stationary part of equations

import numpy as np
import auxillary_functions as func
import constants as co
from scipy import ndimage as sn


#3D solution of stationary equation yielding dT/dt = Rn = T(n+1)-T(n) / delta(t)
def Rn(Tn: np.array, Cn: np.array) -> np.array:
	lap = sn.filters.laplace(Tn)
	watervel = func.u_w(Tn, Cn)
	gradT = np.array(np.gradient(Tn, co.dx))
	stationary = -co.k_m*lap + co.rho_w*co.cp_w*sum(watervel[i] * gradT[i] for i in range(3))
	return -stationary / (co.rho_m*co.cp_m)


#3D solution of stationary equation yielding dC/dt = Sn = C(n+1)-C(n) / delta(t)
def Sn(Tn: np.array, Cn: np.array) -> np.array:
	lap = sn.filters.laplace(Cn)
	v = Cn * func.u_w(Tn, Cn)
	gradT = np.array(np.gradient(Tn, co.dx))
	return co.D * lap - sum(np.gradient(v[i], co.dx)[i] for i in range(3))


def Jacobi(T0: np.array, C0: np.array, steps: int) -> (np.array, np.array):
	#T0, C0 = initial conditions
	T = np.array([np.zeros_like(T0) for i in range(steps+1)])
	C = np.array([np.zeros_like(C0) for i in range(steps+1)])
	T[0] = T0
	C[0] = C0
	for i in range(steps):
		R = Rn(T[i], C[i])
		S = Sn(T[i], C[i])
		T[i+1] = T[i] + co.dt*R
		C[i+1] = C[i] + co.dt*S
	return T,C

### TEST ###
#(ingen grensebetingelser er påført noe sted)#

T0 = np.zeros((5,5,5))
T0[:,0] = 10
C0 = np.ones_like(T0) * 3
C0[0,:] = 0
steps = 1000
T,C = Jacobi(T0, C0, steps)
print(T[0])
print(T[-1])
print(C[0])
print(C[-1])
# Ved å betrakte matrisene ser man at diffusjonen er veldig sakte i forhold til antall tidssteg
