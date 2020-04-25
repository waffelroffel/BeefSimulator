# convergence_test.py

from BeefSimulator import BeefSimulator
import numpy as np
import matplotlib.pyplot as plt
import data_management
from configs.config_library.convergence_test.conf import conf, dt_list, dh_list, dt_default, dh_default
from configs.config_library.convergence_test.Tconf import alp, Lx, Ly, Lz, T


def find_single_dataset_abs_diff( data: np.array, analytical_sol: function, meshg: np.array, time: float ) -> float:
    analytical_data = analytical_sol( meshg[0], meshg[1], meshg[2], time )
    # Frobenius matrix norm
    return = abs(np.linalg.norm(data - analytical_data))


def get_data_from_foldername( foldername: str, time: float ) -> np.array, np.arraym np.array:
    path = Path(foldername)
    head_path = path.joinpath("header.json")
    T_path = path.joinpath("T.dat")
    C_path = path.joinpath("C.dat")

    header = None
    with open(head_path) as f:
        header = json.load(f)

    t_jump = header["t_jump"]
    dt = header["dt"]
    dh = header["dh"]

    dims = header["dims"]
    shape = tuple(header["shape"])
    t = np.linspace(header["t0"], header["tn"], shape[0])
    x = np.linspace(dims["x0"], dims["xn"], shape[1])
    y = np.linspace(dims["y0"], dims["yn"], shape[2])
    z = np.linspace(dims["z0"], dims["zn"], shape[3])

    meshg = np.meshgrid(x, y, z)

    if t_jump == -1:
        t_index = -1
    else:
        t_index = int( (time / dt) / t_jump )

    T_data = np.memmap(T_path, dtype="float64", mode="r", shape=shape)
    C_data = np.memmap(C_path, dtype="float64", mode="r", shape=shape)

    return meshg, T_data[t_index], C_data[T_index]


def du_data( folder_names: list, time: float, analytic: function = T ) -> list:
    l = len(folder_names)
    diff = np.zeros(l)
    for i in range(l):
        meshg, Tdata, Cdata = get_data_from_foldername(folder_names[i], time)
        diff[i] = find_single_dataset_abs_diff(Tdata, analytic, meshg, time)
    return diff


def plot_convergencetest_dt( dt: list, diff: list ) -> None
    plt.plot( dt, diff, marker='o', lw=5 )
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'Error')
    plt.title(r'Error with various $\Delta t$')
    plt.grid()
    plt.show()
    plt.savefig(fname='data/dt_convplot', format='pdf')


def plot_convergencetest_dh( dh: list, diff: list ) -> None
    plt.plot( dh, diff, marker='o', lw=5 )
    plt.xlabel(r'$\Delta h$')
    plt.ylabel(r'Error')
    plt.title(r'Error with various $\Delta h$')
    plt.grid()
    plt.show()
    plt.savefig(fname='data/dh_convplot', format='pdf')


# This only runs if all combinations of dt_list and dh_list convergence tests have been run
if __name__ == '__main__':
    dt_folders = []
    dh_folders = []
    for i in range(len(dt_list)):
        dt_folders.append(f'data/convtest_T_dt{dt_list[i]:.2g}_dh{dh_default:.2g}')
    for j in range(len(dh_lsit)):
        dh_folders.append(f'data/convtest_T_dt{dt_default:.2g}_dh{dh_list[j]:.2g}')

    convtime = conf['tlen']

    diff_t = du_data( dt_folders, convtime, T )
    diff_h = du_data( dh_folders, convtime, T )
    plot_convergencetest_dt( dt_list, diff_t )
    plot_convergencetest_dh( dh_list. diff_h )
