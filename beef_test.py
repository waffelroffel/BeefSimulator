import numpy as np
from BeefSimulator import BeefSimulator
from scipy.sparse import diags


def initial(xx, yy, zz, tt):
    return 1


def a(xx, yy, zz, tt):
    return np.ones(xx.size)


def b(xx, yy, zz, tt):
    return np.ones(xx.size)


def c(xx, yy, zz, tt):
    return np.zeros(xx.size)


def alpha(xx, yy, zz, tt):
    return np.ones(xx.size)


def beta(xx, yy, zz, tt):
    return np.zeros(xx.size)


def gamma(xx, yy, zz, tt):
    return np.zeros(xx.size)


dims = [[0, 20], [0, 20], [0, 20], [0, 3]]
bs = BeefSimulator(dims=dims, a=a, b=b, c=c, alpha=alpha, beta=beta,
                   gamma=gamma, initial=initial, dh=1, dt=0.01, logging=1)
bs.solve_all()
bs.plot(x=10)
