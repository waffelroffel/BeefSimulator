import numpy as np
import constants as const
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

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
f = 0.9
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
    # print(Ceq)
    E = E0 + Em / (1 + np.exp(-En * (T - ED)))      # Size N
    mu_w = np.exp(-0.0072*(T + 273) - 2.8658)       # Size N
    # print(mu_w)
    P = E * (C - Ceq)                               # Size N
    u_w = np.zeros_like(C)
    # Most likely wrong for [0] and [-1]
    u_w[1:-1] = -K*E[1:-1]/mu_w[1:-1] * (P[2:] - P[:-2])
    u_w[0] = u_w[1]
    u_w[-1] = u_w[-2]
    # print(u_w)
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

def FTCS2(T, C, dx, dt):
    """
    :param T: Vector of temperatures, size N
    :param C:  Vector of concentrations, size N
    :param dx: Distance between grid points
    :param dt: Size of time step
    :return Tnew: vector for temperature for the next time step
    :return Cnew: vector for concentration for the next time step
    """
    Ceqfunc = lambda x: a1 - a2 / (1 + a3 * np.exp(-a4 * (x - T_sig))) # Takes in temperature and returns Ceq
    Ceq = Ceqfunc(T)  # Size N
    #print(Ceq)
    E = E0 + Em / (1 + np.exp(-En * (T - ED)))      # Size N
    mu_w = np.exp(-0.0072*T - 2.8658)       # Size N
    # print(mu_w)
    P = E * (C - Ceq)                               # Size N
    u_w = np.zeros_like(C)
    # Most likely wrong for [0] and [-1]
    u_w[1:-1] = -K*E[1:-1]/mu_w[1:-1] * (P[2:] - P[:-2])
    u_w[0] = u_w[1]
    u_w[-1] = u_w[-2]
    u_w[:]=0
    # print(u_w)
    C1 = k_m * dt /(rho_m*cp_m*dx**2)
    C2 = rho_w*cp_w*dt / (rho_m*cp_m *2*dx)
    Tnew = np.zeros_like(T)
    Tnew[1:-1] = (C1 + C2*u_w[1:-1])*T[2:] + (1-2*C1)*T[1:-1] + (C1 - C2*u_w[1:-1])*T[:-2]
    Tnew[0] = 2*C1*T[1] + (1 - 2*C1)*T[0]
    TNp1 = T[-2] - 2*dx*(1-f) * h/k_m*(T_oven - T[-1]) - u_w[-1]*cp_w*rho_w/k_m*T[-1]
    #print("Tnp1 = ", TNp1)
    Tnew[-1] = (C1 + C2*u_w[-1])*TNp1 + (1-2*C1)*T[-1] + (C1 - C2*u_w[-1])*T[-2]
    Cnew = np.zeros_like(C)
    Cnew[1:-1] =  C[1:-1] + D*dt/dx**2 * (C[2:] - 2*C[1:-1] + C[:-2]) - dt/(2*dx) * (u_w[1:-1] * (C[2:] - C[:-2]) + C[1:-1] * (u_w[2:] - u_w[:-2]))
    Cnew[0] = C[0] + D*dt/dx**2 * (C[1] - 2*C[0] + C[1]) #- dt/(2*dx) * (u_w[0] * (C[1] - C[1]) + C[0] * (u_w[1] - u_w[1]))
    CNp1 = C[-2] + 2*dx/D * u_w[-1] * C[-1] - 2*dx/D * f*h * (T_oven - T[-1])/(H_evap*rho_m)*(C[-1] - Ceq[-1])
    Cnew[-1] = C[-1] + h*f*dt/(H_evap*rho_w)*(T_oven - T[-1])*(C[-1] - Ceq[-1])
    return Tnew, Cnew

def FTCS3(T, C, dx, dt):
    """
    :param T: Vector of temperatures, size N
    :param C:  Vector of concentrations, size N
    :param dx: Distance between grid points
    :param dt: Size of time step
    :return Tnew: vector for temperature for the next time step
    :return Cnew: vector for concentration for the next time step
    """
    Ceq_func = lambda x: a1 - a2 / (1 + a3 * np.exp(-a4 * (x - T_sig)))
    mu_w_func = lambda x: np.exp(-0.0072*x - 2.8658)
    E_func = lambda x: E0 + Em / (1 + np.exp(-En * (x - ED)))
    Ceq = Ceq_func(T)
    mu_w = mu_w_func(T)
    E = E_func(T)
    u_w = np.zeros_like(T)
    u_w[1:-1] = -K*E[1:-1]/(mu_w[1:-1]*2*dx) * ((C[2:] - Ceq[2:]) - (C[:-2] - Ceq[:-2]))
    C1 = k_m * dt / (rho_m * cp_m * dx ** 2)
    C2 = rho_w * cp_w * dt / (rho_m * cp_m * 2 * dx)
    Tnew = np.zeros_like(T)
    Cnew = np.zeros_like(T)

    TNp1 = T[-2] - 2 * dx * (1 - f) * h / k_m * (T_oven - T[-1]) - u_w[-1] * cp_w * rho_w / k_m * T[-1]
    Tnew[1:-1] = (C1 + C2 * u_w[1:-1]) * T[2:] + (1 - 2 * C1) * T[1:-1] + (C1 - C2 * u_w[1:-1]) * T[:-2]
    Tnew[0] = 2 * C1 * T[1] + (1 - 2 * C1) * T[0]

    Cnew[1:-1] = C[1:-1] + D * dt / dx ** 2 * (C[2:] - 2 * C[1:-1] + C[:-2]) - dt / (2 * dx) * (
                u_w[1:-1] * (C[2:] - C[:-2]) + C[1:-1] * (u_w[2:] - u_w[:-2]))
    Cnew[0] = C[0] + D * dt / dx ** 2 * (
                C[1] - 2 * C[0] + C[1])  # - dt/(2*dx) * (u_w[0] * (C[1] - C[1]) + C[0] * (u_w[1] - u_w[1]))
    CNp1 = C[-2] + 2 * dx / D * u_w[-1] * C[-1] - 2 * dx / D * f * h * (T_oven - T[-1]) / (H_evap * rho_m) * (
                C[-1] - Ceq[-1])
    return Tnew, Cnew

if __name__ == "__main__":
    N = 101
    x = np.linspace(0, 1, N)
    dx = x[1] - x[0]
    dt = 0.5 * dx**2
    t = np.arange(0, 1 + dt, dt)
    T = np.ones_like(x) * T_0
    C = np.ones_like(x) * C_0
    Ts = np.zeros((len(t), len(x)))
    Cs = np.zeros((len(t), len(x)))
    Ts[0] = T
    Cs[0] = C
    for i in range(1):
        Ts[i+1], Cs[i+1] = FTCS2(Ts[i], Cs[i], dx, dt)
    plt.plot(x, Ts[-1])
    #plt.ylim(0, T_0 * 10)
    #plt.show()
    plt.plot(x, Cs[-1])
    plt.ylim(0, C_0 * 10)
    #plt.show()
    plt.close()