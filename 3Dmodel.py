#3Dmodel.py
#
#Quick and dirty way to solve the stationary part of equations
#TODO: Boundary conditions not implemented

import numpy as np
import auxillary_functions as func
import constants as co
from scipy import ndimage as sn


#3D solution of stationary equation yielding dT/dt = Rn = T(n+1)-T(n) / delta(t)
#NB!!! Highly experimental
def Rn(Tn: np.array, Cn: np.array, dims: int = 3) -> np.array:
	
	#Define quantites for clarity
	lap = sn.filters.laplace(Tn) / (co.dx**2)
	watervel = func.u_w(Tn, Cn)
	gradT = np.array(np.gradient(Tn, co.dx))
	stationary = -co.k_m*lap + co.rho_w*co.cp_w * func.dotND(watervel, gradT)
	
	#Calculate T in the bulk
	T = -stationary / (co.rho_m*co.cp_m)
	
	#Well defined for square beef
	#slice 0 = lowest value ('bottom'), slice -1 = highest value ('top')
	#axis 0 = x, axis 1 = y, axis 2 = z
	#Assumes that T_surf is the unknown in the next time step
	def R_boundary(slice_index: int, axis_index: int):
		if slice_index == 0 or slice_index == -1:
			#Take the correct 2d arrays of relevant quantities
			T_b = Tn.take(indices = slice_index, axis = axis_index)
			grad_b = gradT[axis_index].take(indices = slice_index, axis = axis_index)
			watervel_b = watervel[axis_index].take(indices = slice_index, axis = axis_index)
		else:
			raise IndexError('Accessing the boundary requires a boundary slice index (0 or -1)')
		
		return co.T_oven - 1/((1-co.f)*co.h) * (co.k_m*grad_b + watervel_b*co.cp_w*co.rho_w*T_b)
	
	#Enforce boundary condition with a python hack
	for i in range(dims):
		a = [slice(None)]*T.ndim
		a[i] = 0
		b = [slice(None)]*T.ndim
		b[i] = -1
		# Enforce the calculated boundary condition
		T[tuple(a)] = R_boundary(0,i)
		T[tuple(b)] = R_boundary(-1,i)
		print('xD')
	
	return T


#3D solution of stationary equation yielding dC/dt = Sn = C(n+1)-C(n) / delta(t)
def Sn(Tn: np.array, Cn: np.array) -> np.array:
	lap = sn.filters.laplace(Cn) / (co.dx**2)
	v = Cn * func.u_w(Tn, Cn)
	return co.D * lap - func.div(v, co.dx)


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

T0 = np.ones((5,5,5)) * 25
C0 = np.ones_like(T0) * 3
steps = 5
T,C = Jacobi(T0, C0, steps)
print(T[0][2])
print(T[-1][2])
print(C[0][2])
print(C[-1][2])
# Ved å betrakte matrisene ser man at diffusjonen er veldig sakte i forhold til antall tidssteg
