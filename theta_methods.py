import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter

def Theta_Scheme(theta, D, N, U_old):
    U_new = U_old.copy()
    superDiag = np.zeros(N - 3)
    subDiag = np.zeros(N - 3)
    mainDiag = np.zeros(N - 2)
    D = D*0.1
    RHS = np.zeros(N - 2)

    superDiag[:] = -D*theta
    subDiag[:] = -D*theta
    mainDiag[:] = 1 + 2*theta*D

    RHS[:] = (1 - theta) * D * U_old[:-2] + (1 - 2*D * (1-theta)) * U_old[1:-1] +  (1 - theta) * D * U_old[2:]
    RHS[0] += theta * D * U_old[0]
    RHS[-1] += theta * D * U_old[-1]

    A = scipy.sparse.diags([subDiag, mainDiag, superDiag], [-1, 0, 1], format='csc')
    U_new[1:-1] = scipy.sparse.linalg.spsolve(A, RHS)
    return U_new


if __name__ == '__main__':
    T_L = 50
    T_R = 10
    N = 101
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    t_final = 15
    dt = 0.02
    M = int(t_final/dt)
    fps = 1/dt
    print(fps)
    t = np.linspace(0, t_final, M+1)
    T0 = np.ones_like(x) * T_R
    T0[0] = T_L
    Ts = np.zeros((M+1, N))
    Ts[0] = T0
    D = dt / dx ** 2
    print(D)
    for i in range(len(t)-1):
        T0 = Theta_Scheme(1, D = dt/dx**2, N = N, U_old = T0)
        Ts[i+1] = T0
    '''
    plt.plot(x, Ts[0])
    plt.plot(x, Ts[1])
    plt.plot(x, Ts[6])
    plt.plot(x, Ts[int(3*M/4)])
    plt.plot(x, Ts[-1])
    plt.show()
    '''
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [])
    def init():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 60)
        return ln,

    def update(frame):
        ax.set_title(f'$t =$ {t[frame]:.2f}')
        ln.set_data(x, Ts[frame])
        return ln,

    ani = FuncAnimation(fig, update, repeat=True, init_func=init, frames=M+1)
    ani.save('backward_euler.mp4', fps=fps)


