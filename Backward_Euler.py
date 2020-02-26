import numpy as np
import scipy.linalg as la
import auxillary_functions as af
import constants as co
np.set_printoptions(4)

# TODO
def initialCondTemp(X):
    return X


# TODO
def initialCondWater(X):
    return X


# TODO
def boundaryCond(X):
    return X


# Finito
def blockGenerator(i):

    # C
    B = np.diag(-h + 6 * co.D + 3 * speed[x * y * i: x * y * (i+1)] * h)

    # +x
    k = -co.D + speed[x * y * i: x * y * (i+1) - 1] * h
    k[x-1::x] = [0] * len(k[x-1::x])
    B += np.diag(k, -1)

    # -x
    k = [-co.D] * (np.shape(B)[1] - 1)
    k[x - 1::x] = [0] * len(k[x - 1::x])
    B += np.diag(k, 1)

    # +y
    B += np.diag(-co.D + speed[x * y * i: x * y * (i+1) - x] * h, x)

    # -y
    B += np.diag(-co.D + speed[(x * y * i) + x: x * y * (i+1)] * h, -x)
    return B


# Finito
def matrixGenerator():
    C = -co.D * np.identity(x * y)
    B = blockGenerator(0)
    Q = np.concatenate((B, C, np.zeros((x ** 2, y ** 2))))
    for i in range(1, z-1):
        A = np.diag((-co.D + speed[i * x * y: (i + 1) * x * y]))
        B = blockGenerator(i)
        Q = la.block_diag(Q, np.concatenate((A, B, C)))
    Q = la.block_diag(Q, np.concatenate((np.zeros((x ** 2, y ** 2)), A, B)))
    return Q

# TODO
def targetGenerator():
    return

"""
Solve system:
[[B C 0 .           0]
 ...
 [0 .. 0 A B C 0 .. 0]
 ...
 [0       ..    0 A B]]
"""


# dimensions

x = 3
y = 3
z = 3
t = 3
h = 0.1

# main

Tn = initialCondTemp(np.zeros(x * y * z))
Cn = initialCondWater(np.zeros(x * y * z))
speed = af.u_w(Tn, Cn)
print(matrixGenerator())

# for i in range(t):
#    speed = af.u_w


