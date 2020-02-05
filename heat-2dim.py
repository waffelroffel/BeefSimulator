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
        return 1#np.cos(t)

    def u_y0(t):
        return 1#np.sin(t)

    def u_y1(t):
        return 1#np.cos(t)

    # initial values
    def t_0(x, y):
        return 0

    # Exact solution
    #def u_exact(x, t):
    #    return np.exp(-np.pi**2*t)*np.cos(np.pi*x)

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
    t_len= 100
    t_steps = int((t_len-t_start)/dt)

    x = np.linspace(x_start, x_len-x_start, x_steps,dtype=np.int)
    y = np.linspace(y_start, y_len-y_start, y_steps,dtype=np.int)
    t = np.linspace(t_start, t_len-t_start, t_steps)

    # Array to store the solution
    U = np.zeros((x_steps, y_steps, t_steps))
    U[:, :, 0] = t_0(x,y)
    U[:,0,:]= u_x0(t)
    U[:,-1,:]= u_x1(t)
    U[0,:,:]= u_y0(t)
    U[-1,:,:]= u_y1(t)


    """
    x -> i
    y -> j
    t -> k
    """
    #print(U[:,:,0])
    for kk in t:
        k = np.where(t==kk)[0][0]
        print(k)
        for j in y[1:-1]:
            for i in x[1:-1]:
                U[i,j,k+1] = U[i,j,k] + dt*alpha*(U[i,j-1,k]+U[i-1,j,k]-4*U[i,j,k]+U[i+1,j,k]+U[i,j+1,k])/h**2 #(dx=dy)
        if k % 20 == 0:
            plt.imshow(U[:,:,k])
            plt.show()

heat_equation2D()
