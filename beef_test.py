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


dh = 1
dt = 0.001

dims = [[0, 4], [0, 4], [0, 2], [0, 1]]
shape = [ int((dims[0][1]-dims[0][0])/dh), int((dims[1][1]-dims[1][0])/dh), int((dims[2][1]-dims[2][0])/dh) ]
bs = BeefSimulator(dims=dims, a=a, b=b, c=c, alpha=alpha, beta=beta,
                   gamma=gamma, initial=initial_T, initial_C=initial_C, dh=dh, dt=dt,
				   logging=0, bnd_types=["d", "d", "d", "d", "d", "d"])

bs.solve_all()
bs.plot(x=10)

x = np.linspace(0, 4, 5)
y = np.linspace(0, 4, 5)
z = np.linspace(0, 2, 3)
t = 1
xx, yy, zz = np.meshgrid(x, y, z)

analytic = T(xx, yy, zz, t).flatten()
i = 100
j = 110
print(bs.T0)
print(analytic)
