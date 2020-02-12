import numpy as np
import constants as const
import matplotlib.pyplot as plt

k_m = const.k_m
rho_m = const.rho_m
cp_m = const.cp_m
rho_w = const.rho_w
cp_w = const.cp_w
D = const.D
a1 = const.a1
a2 = const.a2
a3 = const.a3
a4 = const.a4
T_sig = const.T_sig
E0 = const.E0
Em = const.Em
En = const.En
ED = const.ED
K = const.K
f = 0.5
h = const.h
T_oven = const.T_oven
H_evap = const.H_evap
T_0 = const.T_0
C_0 = const.C_0


def FTCS(T, C, dx, dt):
    """
    :param T: Vector of temperatures, size N
    :param C:  Vector of concentrations, size N
    :param dx: Distance between grid points
    :param dt: Size of timestep
    :return: Tnew, Cnew. vectors for temperature and concentration for the next timestep
    """
    Ceq = a1 - a2 / (1 + a3*np.exp(-a4*(T-T_sig)))  # Size N
    #print(Ceq)
    E = E0 + Em / (1 + np.exp(-En * (T - ED)))      # Size N
    mu_w = np.exp(-0.0072*(T + 273) - 2.8658)               # Size N
    #print(mu_w)
    P = E * (C - Ceq)                               # Size N
    u_w = np.zeros_like(C)
    # Most likely wrong for [0] and [-1]
    u_w[1:-1] = -K*E[1:-1]/mu_w[1:-1] * (P[2:] - P[:-2])
    u_w[0] = u_w[1]
    u_w[-1] = u_w[-2]
    #print(u_w)
    C1 = k_m * dt /(rho_m*cp_m*dx**2)
    C2 = rho_w*cp_w*dt / (rho_m*cp_m *2*dx * u_w[1:-1]+0.0001)
    Tnew = np.zeros_like(T)
    Tnew[1:-1] = (1 - 2*C1) * T[1:-1] + (C1 - C2)*T[2:] + (C1+ C2)*T[:-2]
    Tnew[0] = Tnew[1]
    Tnew[-1] = T[-1] + dt/(rho_m*cp_m)* (1-f)*h*(T_oven - T[-1])
    Cnew = np.zeros_like(C)
    Cnew[1:-1] = (1 - 2*D*dt/dx**2 - dt / (2*dx)*(u_w[2:]- u_w[0:-2]))*C[1:-1] + (D*dt/dx**2 - u_w[1:-1]*dt/(2*dx))*C[2:] + (D*dt/dx**2 + u_w[1:-1]*dt/(2*dx))*C[:-2]
    Cnew[0] = Cnew[1]
    Cnew[-1] = C[-1] + h*f*dt/(H_evap*rho_w)*(T_oven - T[-1])*(C[-1] - Ceq[-1])
    return Tnew, Cnew


if __name__ == "__main__":
    N = 11
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    dt = 0.05 * dx**2
    t = np.arange(0, 1 + dt, dt)
    T = np.ones_like(x) * T_0
    C = np.ones_like(x) * C_0
    Ts = np.zeros((len(t), len(x)))
    Cs = np.zeros((len(t), len(x)))
    Ts[0] = T
    Cs[0] = C
    for i in range(len(t)-1):
        Ts[i+1], Cs[i+1] = FTCS(Ts[i], Cs[i], dx, dt)
    plt.plot(x, Ts[0])
    plt.show()
    plt.plot(x, Cs[0])
    plt.show()