import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

L = 0.1
n = 11 # Includes boundaries
dx = L/n

T0 = 0.
T1 = 45.0
T2 = 15.0

alpha = 0.0001
beta = 0.005

fps = 60
dt = 0.05
t_final = 30
t_steps = int(t_final / dt)
t = np.arange(0, t_final, dt)

x = np.linspace(0, L, n)

T_t1 = np.ones((len(t), n)) * T0
T_t1[:,0] = T1
T_t1[:,-1] = T2

T_t2 = np.ones((len(t), n)) * T0
T_t2[:,0] = T1
T_t2[:,-1] = T2

T = np.ones(n) * T0
T[0] = T1
T[-1] = T2
dTdt = np.empty(n)

for i in range(t_steps):
    dTdt[1:-1] = alpha * ( (T[0:-2] - 2 * T[1:-1] + T[2:]) / dx**2)
    T[1:-1] += dTdt[1:-1] * dt
    T_t1[i, 1:-1] = T[1:-1]

T = np.ones(n) * T0
T[0] = T1
T[-1] = T2
dTdt = np.empty(n)

for i in range(t_steps):
    dTdt[1:-1] = alpha * ( (T[0:-2] - 2 * T[1:-1] + T[2:]) / dx**2) + beta * (T[0:-2] - T[1:-1]) / dx
    T[1:-1] += dTdt[1:-1] * dt
    T_t2[i, 1:-1] = T[1:-1]

fig, ax = plt.subplots()

line1, = ax.plot([], [], lw=.75, color='r', label=r'$\partial_t T = - \alpha \partial_{xx} T$')
line2, = ax.plot([], [], lw=.75, color='b', label=r'$\partial_t T = - \alpha \partial_{xx} T + \beta \partial_{x} T$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$T(x,t)$')
ax.set_xlim(left = 0, right = L)
ax.set_ylim(bottom = 0, top = 50)
ax.legend()
ax.grid()

def animate(i):
    print(i + 1, "out of", t_steps)

    ax.set_title(r'$\alpha =$ ' + str(alpha) + r';$\beta =$ ' + str(beta) + f';$t =$ {i*dt:.2f}')
    line1.set_data(x, T_t1[i])
    line2.set_data(x, T_t2[i])

    return [line1, line2]

anim = animation.FuncAnimation(fig, animate, repeat=False, frames=int(t_steps),
                                    interval=1000 / fps, blit=False)
filename = "heat_eq_time-evo.mp4"

anim.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'], dpi=200, bitrate=-1)
