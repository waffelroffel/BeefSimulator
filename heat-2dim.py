import numpy as np
from scipy.sparse import diags
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # For 3-d plott
from matplotlib import cm
from time import sleep
import matplotlib.animation as animation
plt.rcParams['axes.grid'] = True


def heat_equation2D():
    # Apply implicit Euler and Crank-Nicolson on
    # the heat equation u_t= a*(u_xx + u_yy)

    # Boundary values
    def u_x0(t):
        return 1  # np.sin(t)

    def u_x1(t):
        return 0  # np.cos(t)

    def u_y0(t):
        return 0.5  # np.sin(t)

    def u_y1(t):
        return 0.25  # np.cos(t)

    # initial values
    def t_0(x, y):
        return 0

    alpha = 1
    h = 1
    x_start = 0
    x_len = 100
    y_start = 0
    y_len = 150

    x_steps = int((x_len-x_start)/h)
    y_steps = int((y_len-y_start)/h)

    dt = h**2/(4*alpha)
    t_start = 0
    t_len = 50
    t_steps = int((t_len)/dt) + 1

    x = np.linspace(x_start, x_len-x_start, x_steps, dtype=np.int)
    y = np.linspace(y_start, y_len-y_start, y_steps, dtype=np.int)
    t = np.linspace(t_start, t_len-t_start, t_steps)

    # Animation settings
    fps = 60

    # Array to store the solution
    U = np.zeros((y_steps, x_steps, t_steps))
    print(U.size)
    U[:, :, 0] = t_0(x, y)
    U[0, :, :] = u_x0(t)
    U[-1, :, :] = u_x1(t)
    U[:, 0, :] = u_y0(t)
    U[:, -1, :] = u_y1(t)

    print("Calculating U(x,y,t)")
    for i, kk in enumerate(t[:-1]):
        # print(i)
        U[1:-1, 1:-1, i + 1] = U[1:-1, 1:-1, i] \
            + dt*alpha*(U[1:-1, 0:-2, i]
                        + U[0:-2, 1:-1, i]
                        - 4*U[1:-1, 1:-1, i]
                        + U[2:, 1:-1, i]
                        + U[1:-1, 2:, i])/h**2
    print("Done iterating!")

    xv, yv = np.meshgrid(x, y)
    fig, ax = plt.subplots()

    cs1 = [ax.contourf(xv, yv, U[:, :, 0], 65, cmap=cm.magma)]
    cbar1 = fig.colorbar(cs1[0], ax=ax, shrink=0.9)
    cbar1.ax.set_ylabel(r'$U(x,y)$', fontsize=14)
    plt.xlabel(r"$x$", fontsize=16)
    plt.ylabel(r"$y$", fontsize=16)
    plt.tight_layout()
    plt.show()

    def animate(i):
        print(i + 1, "out of", t_steps)

        for tp in cs1[0].collections:
            tp.remove()

        ax.set_title(r'$\alpha =$ ' + str(alpha) + f';$t =$ {i*dt:.2f}')
        cs1[0] = ax.contourf(xv, yv, U[:, :, i], cmap=cm.magma)
        cbar1 = fig.colorbar(cs1[0], ax=ax, shrink=0.9)
        return cs1[0]

    print("Animating...")
    anim = animation.FuncAnimation(fig, animate, repeat=False, frames=int(t_steps),
                                   interval=1000 / fps, blit=False)
    filename = "2D-heateq.mp4"
    print("Done!")
    anim.save(filename, fps=fps, extra_args=[
              '-vcodec', 'libx264'], dpi=200, bitrate=-1)


heat_equation2D()
