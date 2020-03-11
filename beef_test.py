import numpy as np
from BeefSimulator import BeefSimulator


def x0(t):
    return np.sin(t)


def x1(t):
    return np.sin(t)


def y0(t):
    return np.cos(t)


def y1(t):
    return np.cos(t)

# initial values


def t0(x, y):
    return 0  # np.sin(x)**2+np.cos(y)**2


dims = [[0, 9], [0, 3], [0, 4], [0, 9]]
conds = [t0, x0, x1, y0, y1]
bs = BeefSimulator(dims=dims, a=1, b=0, c=1, alpha=1, beta=1,
                   gamma=1, initial=0, dh=1, dt=1, logging=1)
bs.solve_next()
print(bs.T1)
