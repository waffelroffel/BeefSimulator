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


dims = [[0, 50], [50, 100], [0, 40]]
conds = [t0, x0, x1, y0, y1]
bf = BeefSimulator(dims, 0.5, 1, conds)
bf.apply_conditions()
bf.solve_all()
bf.plot()
