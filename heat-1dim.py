import numpy as np
from scipy.sparse import diags
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # For 3-d plott
from matplotlib import cm
plt.rcParams['axes.grid'] = True

# Numerical differentiation

# Forward difference


def diff_forward(f, x, h=0.1):
    return (f(x+h)-f(x))/h


def diff_backward(f, x, h=0.1):
    return (f(x)-f(x-h))/h


def diff_central(f, x, h=0.1):
    return (f(x+h)-f(x-h))/(2*h)


def diff2_central(f, x, h=0.1):
    return (f(x+h)-2*f(x)+f(x-h))/h**2


def test_diffs():
    x = np.pi/4
    df_exact = np.cos(x)
    ddf_exact = -np.sin(x)
    h = 0.1
    f = np.sin
    df = diff_forward(f, x, h)
    print('Approximations to the first derivative')
    print('Forward difference:  df = {:12.8f},   Error = {:10.3e} '.format(
        df, df_exact-df))
    df = diff_backward(f, x, h)
    print('Backward difference: df = {:12.8f},   Error = {:10.3e} '.format(
        df, df_exact-df))
    df = diff_central(f, x, h)
    print('Central difference:  df = {:12.8f},   Error = {:10.3e} '.format(
        df, df_exact-df))
    print('Approximation to the second derivative')
    ddf = diff2_central(f, x, h)
    print('Central difference:  ddf= {:12.8f},   Error = {:10.3e} '.format(
        ddf, ddf_exact-ddf))


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


def heat_equation():
    # Apply implicit Euler and Crank-Nicolson on
    # the heat equation u+t=u_{xx}

    # Define the problem of example 3
    def f3(x):
        return np.cos(np.pi*x)

    # Boundary values
    def g0(t):
        return np.exp(-np.pi**2*t)

    def g1(t):
        return -np.exp(-np.pi**2*t)

    # Exact solution
    def u_exact(x, t):
        return np.exp(-np.pi**2*t)*np.cos(np.pi*x)

    f = f3

    # Choose method
    #method = 'iEuler'
    method = 'CrankNicolson'

    M = 100                   # Number of intervals in the x-direction
    Dx = 1/M
    x = np.linspace(0, 1, M+1)   # Gridpoints in the x-direction

    tend = 0.5
    N = M  # Number of intervals in the t-direction
    Dt = tend/N
    t = np.linspace(0, tend, N+1)  # Gridpoints in the t-direction

    # Array to store the solution
    U = np.zeros((M+1, N+1))
    U[:, 0] = f(x)              # Initial condition U_{i,0} = f(x_i)

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


heat_equation()
