import numpy as np
from scipy.sparse import diags
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # For 3-d plott
from matplotlib import cm
plt.rcParams['axes.grid'] = True

# Numerical differentiation

# Forward difference


def diff_forward(f, x, y, hx=0.1, hy=0.1):
    pass


def diff_backward(f, x, y, hx=0.1, hy=0.1):
    pass


def diff_central(f, x, y, hx=0.1, hy=0.1):
    pass


def diff2_central(f, x, y, hx=0.1, hy=0.1):
    pass


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
    # the heat equation u+t=u_{xx}

    # Define the problem of example 3
    # Boundary values
    def u_x0(t):
        return np.sin(t)

    def u_x1(t):
        return np.cos(t)

    def u_y0(t):
        return np.sin(t)

    def u_y1(t):
        return np.cos(t)

    def t_0(x, y):
        return x+y

    # Exact solution
    def u_exact(x, t):
        return np.exp(-np.pi**2*t)*np.cos(np.pi*x)

    x_start = 0
    x_end = 100
    x_steps = 100

    y_start = 0
    y_end = 100
    y_steps = 100

    t_start = 0
    t_end = 100
    t_steps = 100

    alpha=1 

    x = np.linspace(x_start, x_end, x_steps,dtype=np.int)
    y = np.linspace(y_start, y_end, y_steps,dtype=np.int)
    t = np.linspace(t_start, t_end, t_steps,dtype=np.int)

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dt = t[1]-t[0]

    # Array to store the solution
    U = np.zeros((x_steps, y_steps, t_steps))
    U[:,0,:]=u_x0(t)
    U[:,-1,:]=u_x1(t)
    U[0,:,:]=u_y0(t)
    U[-1,:,:]=u_y1(t)
    U[:, :, 0] = t_0(x,y)


    """
    x -> i
    y -> j
    t -> k
    """
    print(U[:,:,0])
    for k in t:
        for j in y[1:-1]:
            for i in x[1:-1]:
                U[i,j,k+1] = U[i,j,k] + dt*alpha*(U[i,j-1,k]+U[i-1,j,k]+U[i+1,j,k]+U[i,j+1,k])/dx**2 #(dx=dy)
        break

    print(U[:,:,1])
    """
    # Set up the matrix K:
    A = tridiag(1, -2, 1, M-1)
    r = Dt/Dx**2
    print('r = ', r)
    if method is 'iEuler':
        K = np.eye(M-1) - r*A
    elif method is 'CrankNicolson':
        K = np.eye(M-1) - 0.5*r*A
    
    Utmp = U[1:-1, 0]          # Temporary solution for the inner gridpoints.

    # Main loop over the time steps.
    for n in range(N):
        # Set up the right hand side of the equation KU=b:
        if method is 'iEuler':
            b = np.copy(Utmp)                   # NB! Copy the array
            b[0] = b[0] + r*g0(t[n+1])
            b[-1] = b[-1] + r*g1(t[n+1])
        elif method is 'CrankNicolson':
            b = np.dot(np.eye(M-1)+0.5*r*A, Utmp)
            b[0] = b[0] + 0.5*r*(g0(t[n])+g0(t[n+1]))
            b[-1] = b[-1] + 0.5*r*(g1(t[n])+g1(t[n+1]))

        Utmp = solve(K, b)         # Solve the equation K*Utmp = b

        U[1:-1, n+1] = Utmp        # Store the solution
        U[0, n+1] = g0(t[n+1])    # Include the boundaries.
        U[M, n+1] = g1(t[n+1])
# end of use the implicit methods
    
    plot_heat_solution(x, t, U)

    # Plot the error if the exact solution is available
    T, X = np.meshgrid(t, x)
    error = u_exact(X, T) - U
    plot_heat_solution(x, t, error, txt='Error')
    print('Maximum error: {:.3e}'.format(max(abs(error.flatten()))))
# end of heat_eqation
    """


heat_equation2D()
