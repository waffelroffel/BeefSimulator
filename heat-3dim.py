import numpy as np
from scipy.sparse import diags
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # For 3-d plott
from matplotlib import cm
from time import sleep
import matplotlib.animation as animation
plt.rcParams['axes.grid'] = True


def heat_equation3D():
    # central diff
    # the heat equation u_t= a*(u_xx + u_yy + u_zz)

    # Boundary values
    def u_x0(t):
        return 1  # np.sin(t)

    def u_x1(t):
        return 1  # np.cos(t)

    def u_y0(t):
        return 1  # np.sin(t)

    def u_y1(t):
        return 1  # np.cos(t)

    def u_z0(t):
        return 1  # np.sin(t)

    def u_z1(t):
        return 1  # np.cos(t)

    # initial values
    def t_0(x, y, z):
        return 0

    alpha = 1
    h = 1
    x_start = 0
    x_len = 100
    y_start = 0
    y_len = 50
    z_start = 0
    z_len = 1

    x_steps = int((x_len-x_start)/h)
    y_steps = int((y_len-y_start)/h)
    z_steps = int((z_len-z_start)/h)

    dt = h**2/(4*alpha)
    t_start = 0
    t_len = 100
    t_steps = int((t_len)/dt) + 1

    x = np.linspace(x_start, x_len-x_start, x_steps, dtype=np.int)
    y = np.linspace(y_start, y_len-y_start, y_steps, dtype=np.int)
    z = np.linspace(z_start, z_len-z_start, z_steps, dtype=np.int)
    t = np.linspace(t_start, t_len-t_start, t_steps)

    # Array to store the solution
    U = np.zeros((x_steps, y_steps, z_steps, t_steps))
    print(U.size)
    U[:, :, :, 0] = t_0(x, y, z)
    U[0, :, :, :] = u_x0(t)
    U[-1, :, :, :] = u_x1(t)
    U[:, 0, :, :] = u_y0(t)
    U[:, -1, :, :] = u_y1(t)
    U[:, :, 0, :] = u_z0(t)
    U[:, :, -1, :] = u_z1(t)

    # print(U[:,:,0])
    mu = dt*alpha/h**2
    print(U[:, :, ])

    for i, kk in enumerate(t[:-1]):
        U[1:-1, 1:-1, 1:-1, i + 1] = U[1:-1, 1:-1, 1:-1, i] \
            + mu*(U[2:, 1:-1, 1:-1, i]
                  + U[:-2, 1:-1, 1:-1, i]
                  + U[1:-1, 2:, 1:-1, i]
                  + U[1:-1, :-2, 1:-1, i]
                  + U[1:-1, 1:-1, 2:, i]
                  + U[1:-1, 1:-1, :-2, i]
                  - 6*U[1:-1, 1:-1, 1:-1, i])
        plt.imshow(U[:, :, 0, i+1])
        print(U[:, 1:, 0, i+1])
        plt.show()


heat_equation3D()
