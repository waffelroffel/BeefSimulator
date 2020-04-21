from pathlib import Path
import numpy as np
import json
from numpy.linalg import norm
from configs.verify_decoupled.T_conf import T_analytic
from configs.verify_decoupled.C_conf import C_analytic
import matplotlib.pyplot as plt


def plot_scalar_error(id, numerical_data, analytic_f, mesh, t_linscape, dt, t_jump=1):
    MSE = []
    print(t_linscape)
    for t, n in zip(tt, (tt/(dt*t_jump)).round().astype(int)):
        numerical = numerical_data[n]
        analytical = analytic_f(None, None, None, *mesh, t)
        MSE.append(np.mean((numerical-analytical)**2))
    plt.plot(tt, MSE)
    plt.xlabel("t")
    plt.ylabel("error")
    plt.title(f'{id}: error in t')
    plt.show()


if __name__ == "__main__":
    folder = "verify_decoupled"
    path = Path("data", folder)
    head_path = path.joinpath("header.json")
    temp_path = path.joinpath("T.dat")
    cons_path = path.joinpath("C.dat")

    header = None
    with open(head_path) as f:
        header = json.load(f)

    t_jump = header["t_jump"]
    dt = header["dt"]
    dh = header["dh"]

    dims = header["dims"]
    shape = tuple(header["shape"])
    tt = np.linspace(header["t0"], header["tn"], shape[0])
    x = np.linspace(dims["x0"], dims["xn"], shape[1])
    y = np.linspace(dims["y0"], dims["yn"], shape[2])
    z = np.linspace(dims["z0"], dims["zn"], shape[3])
    mesh = np.meshgrid(x, y, z)

    T_data = np.memmap(
        temp_path, dtype="float64", mode="r", shape=shape)
    C_data = np.memmap(
        cons_path, dtype="float64", mode="r", shape=shape)

    plot_scalar_error("Temperature", T_data, T_analytic, mesh, tt, dt, t_jump)
    plot_scalar_error("Concentration", C_data,
                      C_analytic, mesh, tt, dt, t_jump)
