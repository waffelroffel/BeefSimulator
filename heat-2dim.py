import numpy as np
from scipy.sparse import diags
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # For 3-d plott
from matplotlib import cm
from time import sleep
import matplotlib.animation as animation
plt.rcParams['axes.grid'] = True

# Numerical differentiation

# Forward difference

def diff_forward(f, x, y, hx=0.1, hy=0.1):
    pass


def diff_backward(f, x, y, hx=0.1, hy=0.1):
    pass


def diff_central(f, x, h=0.1):
    return (f(x+h)-f(x-h))/(2*h)


def diff2_central(f, x, h=0.1):
    return (f(x+h)-2*f(x)+f(x-h))/h**2


def test_diffs():
    pass


def tridiag(v, d, w, N):
    # Help function
    # Returns a tridiagonal matrix A=tridiag(v, d, w) of dimension N x N.
    e = np.ones(N)        # array [1,1,...,1] of length N
    A = v*np.diag(e[1:], -1)+d*np.diag(e)+w*np.diag(e[1:], 1)
    return A


# The heat equation (time dependent PDEs)
# ---------------------------------------

def plot_heat_solution(x, t, U, txt='Solution'):
    # Help function
    # Plot the solution of the heat equation
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    T, X = np.meshgrid(t, x)
    ax.plot_wireframe(T, X, U)
    ax.plot_surface(T, X, U)
    ax.view_init(azim=30)              # Rotate the figure
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(txt)
    plt.show()
# end of plot_heat_solution


def heat_equation2D():
    # Apply implicit Euler and Crank-Nicolson on
    # the heat equation u_t= a*(u_xx + u_yy)

    # Boundary values
    def u_x0(t):
        return 1#np.sin(t)

    def u_x1(t):
        return 0#np.cos(t)

    def u_y0(t):
        return 0.5#np.sin(t)

    def u_y1(t):
        return 0.25#np.cos(t)

    # initial values
    def t_0(x, y):
        return 0

    alpha=1
    h = 1
    x_start = 0
    x_len = 100
    y_start = 0
    y_len = 150

    x_steps = int((x_len-x_start)/h)
    y_steps = int((y_len-y_start)/h)

    dt = h**2/(4*alpha)
    t_start = 0
    t_len= 50
    t_steps = int((t_len)/dt) + 1

    x = np.linspace(x_start, x_len-x_start, x_steps,dtype=np.int)
    y = np.linspace(y_start, y_len-y_start, y_steps,dtype=np.int)
    t = np.linspace(t_start, t_len-t_start, t_steps)

    # Animation settings
    fps = 60

    # Array to store the solution
    U = np.zeros((y_steps, x_steps, t_steps))
    print(U.size)
    U[:, :, 0] = t_0(x,y)
    U[0,:,:]= u_x0(t)
    U[-1,:,:]= u_x1(t)
    U[:,0,:]= u_y0(t)
    U[:,-1,:]= u_y1(t)


    """
    x -> i
    y -> j
    t -> k
    """

    print("Calculating U(x,y,t)")
    for i,kk in enumerate(t[:-1]):
        #print(i)
        U[1:-1, 1:-1, i + 1] = U[ 1:-1, 1:-1, i] \
                               + dt*alpha*(U[ 1:-1, 0:-2, i]
                                           + U[ 0:-2, 1:-1, i]
                                           - 4*U[ 1:-1, 1:-1, i]
                                           + U[ 2:, 1:-1, i]
                                           + U[ 1:-1, 2:, i])/h**2
    print("Done!")

    print(U[:,:,0].shape)

    xv, yv = np.meshgrid(x,y)
    fig, ax = plt.subplots()

    cs1 = [ax.contourf(xv, yv, U[:,:,0], 65, cmap = cm.magma)]
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
        cs1[0] = ax.contourf(xv, yv, U[:,:,i], cmap = cm.magma)
        cbar1 = fig.colorbar(cs1[0], ax=ax, shrink=0.9)
        return cs1[0]

    print("Animating...")
    anim = animation.FuncAnimation(fig, animate, repeat=False, frames=int(t_steps),
                                   interval=1000 / fps, blit=False)
    filename = "2D-heateq.mp4"
    print("Done!")
    anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'], dpi=200, bitrate=-1)

heat_equation2D()
