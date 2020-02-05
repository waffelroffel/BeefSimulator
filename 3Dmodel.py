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

#TODO: implement same for dC/dt
