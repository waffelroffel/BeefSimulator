import numpy as np
from BeefSimulator import BeefSimulator
from scipy.sparse import diags


def T(x, y, z, t):
    return 3 * np.exp(-4*1e-3*(np.pi)**2*(1+1+4) * t) * \
        np.sin(2*np.pi * x) * np.sin(2*np.pi * y) * np.sin(4*np.pi * z)


def initial_T(xx, yy, zz, tt):
    return T(xx, yy, zz, tt)


def initial_C(xx, yy, zz, tt):
	return 1


def a(xx, yy, zz, tt):
    return np.ones(xx.size)*1e-3


def b(xx, yy, zz, tt):
    return np.ones(xx.size)


def c(xx, yy, zz, tt):
    return np.zeros(xx.size)


def alpha(xx, yy, zz, tt):
    return np.zeros(xx.size)


def beta(xx, yy, zz, tt):
    return np.ones(xx.size)


def gamma(xx, yy, zz, tt):
    return np.zeros(xx.size)

# This diverges even for very low values of dt :(
dh = 0.01
dt = 0.0000001

dims = [[0, 1], [0, 1], [0, 1], [0, 0.00001]]
shape = [ int((dims[0][1]-dims[0][0])/dh), int((dims[1][1]-dims[1][0])/dh), int((dims[2][1]-dims[2][0])/dh) ]
bs = BeefSimulator(dims=dims, a=a, b=b, c=c, alpha=alpha, beta=beta,
                   gamma=gamma, initial=initial_T, initial_C=initial_C, dh=dh, dt=dt,
				   logging=2, bnd_types=["d", "d", "d", "d", "d", "d"])
bs.solve_all()

bs.plot(0.000001, z=0.5)
bs.plot(0.000002, z=0.5)
bs.plot(0.000003, z=0.5)
bs.plot(0.000004, z=0.5)
bs.plot(0.000005, z=0.5)
bs.plot(0.000006, z=0.5)
bs.plot(0.000007, z=0.5)
bs.plot(0.000008, z=0.5)
bs.plot(0.000009, z=0.5)



'''
x = np.linspace(0, 3, 5)
y = np.linspace(0, 2, 5)
z = np.linspace(0, 1, 3)
t = 1
xx, yy, zz = np.meshgrid(x, y, z)

analytic = T(xx, yy, zz, t).flatten()
i = 100
j = 110
print(bs.T0)
print(analytic)
'''