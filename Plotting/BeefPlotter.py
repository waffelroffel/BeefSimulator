import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
from matplotlib import rc
from pathlib import Path
from typing import Union
import json


def init_3d(fig, axes):
    axes.append(fig.gca(projection="3d"))
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_zlabel("$z$")


class Plotter:
    MODES = {"S", "M"}
    IDS = {"T", "C"}
    CMAPS = {"T": cm.get_cmap("magma"),
             "C": cm.get_cmap("viridis")}
    TYPES = {"T": "Temperature",
             "C": "Concentration"}

    def __init__(self, beefsim=None, name="untitled", save_fig=False):
        """
        beefsim: A BeefSimulator object with axis and stepping data for plotting.
        name: Filename for saved plots. Default: "untitled"
        save_fig: Determines whether the plots will be saved. Default: "False"
        """
        if beefsim is None:
            self.load_from_file(name)
        else:
            self.load_from_class(beefsim)

        levels_T = np.linspace(self.vmin_T, self.vmax_T, 65)
        levels_C = np.linspace(self.vmin_C, self.vmax_C, 65)
        self.LEVELS = {"T": levels_T, "C": levels_C}
        self.name = name
        self.save_fig = save_fig

        self.MODES = {"S": self.singlecross,
                      "M": self.multicross}

    def load_from_class(self, beefsim):
        self.t = beefsim.t
        self.x = beefsim.x
        self.y = beefsim.y
        self.z = beefsim.z
        self.t_jump = beefsim.t_jump
        self.dt = beefsim.dt
        self.dh = beefsim.dh
        # TODO: not guaranteed to get min max from first timestep
        # np.min(beefsim.T0), np.max(beefsim.T0)
        self.vmin_T, self.vmax_T = 15, 25
        # np.min(beefsim.C0), np.max(beefsim.C0)
        self.vmin_C, self.vmax_C = 15, 25

    def load_from_file(self, path: Path):
        if not isinstance(path, Path):
            path = Path(path)
        head_path = path.joinpath("header.json")
        temp_path = path.joinpath("T.dat")
        cons_path = path.joinpath("C.dat")

        header = None
        with open(head_path) as f:
            header = json.load(f)

        self.t_jump = header["t_jump"]
        self.dt = header["dt"]
        self.dh = header["dh"]

        dims = header["dims"]
        shape = tuple(header["shape"])
        self.t = np.linspace(header["t0"], header["tn"], shape[0])
        self.x = np.linspace(dims["x0"], dims["xn"], shape[1])
        self.y = np.linspace(dims["y0"], dims["yn"], shape[2])
        self.z = np.linspace(dims["z0"], dims["zn"], shape[3])

        self.T_data = np.memmap(
            temp_path, dtype="float64", mode="r", shape=shape)
        self.C_data = np.memmap(
            cons_path, dtype="float64", mode="r", shape=shape)
        # TODO: need better selection of min/max
        self.vmin_T = 0
        self.vmax_T = np.max(self.T_data)
        self.vmin_C = 0
        self.vmax_C = np.max(self.C_data)

    def show_heat_map2(self, id, T, X=[], Y=[], Z=[]):
        U = self.T_data if id == "T" else self.C_data
        self.show_heat_map(U, id, T, X, Y, Z)

    def show_heat_map(self, U, id, T, X=[], Y=[], Z=[]):
        if id not in self.IDS:
            raise ValueError(
                "Trying to aquire a quantity that does not exist.")
        mode, extra = self.get_mode(X, Y, Z)

        if mode not in self.MODES:
            raise Exception("Invalid mode given")

        for t, n in self.index_t(T):
            if mode == "S":
                self.singlecross(U, t, n, extra[0], extra[1], extra[2], id)
            elif mode == "M":
                self.multicross(U, t, n, X, Y, Z, id)
            else:
                raise Exception("Mode not implemented!")

    def multicross(self, U, t, n, X, Y, Z, id):
        yz, zy = np.meshgrid(self.y, self.z, indexing="ij")
        xz, zx = np.meshgrid(self.x, self.z, indexing="ij")
        xy, yx = np.meshgrid(self.x, self.y, indexing="ij")

        fig = plt.figure()
        axes = []
        cs = []
        cbarlab = ""
        coordlab = ""

        init_3d(fig, axes)
        axes[0].text2D(
            0.5, 0.95, f'{self.TYPES[id]} distribution @ $t=$ {t: .3g}', transform=axes[0].transAxes)

        if (not Y and not Z):
            axes[0].view_init(15, - 107)
        elif (not X and not Z):
            axes[0].view_init(15, - 16)
        elif (not X and not Y):
            axes[0].view_init(12, - 30)

        axes[0].set_xlim3d(self.x[0] if not X else X[0],
                           self.x[-1] if not X else X[-1])
        axes[0].set_ylim3d(self.y[0] if not Y else Y[0],
                           self.y[-1] if not Y else Y[-1])
        axes[0].set_zlim3d(self.z[0] if not Z else Z[0],
                           self.z[-1] if not Z else Z[-1])

        for x, i in self.index_h(X):
            cs.append(axes[0].contourf(U[n, i, :, :], yz, zy,
                                       zdir="x", offset=x, levels=self.LEVELS[id], cmap=self.CMAPS[id]))
        for y, j in self.index_h(Y):
            cs.append(axes[0].contourf(xz, U[n, :, j, :], zx,
                                       zdir="y", offset=y, levels=self.LEVELS[id], cmap=self.CMAPS[id]))
        for z, k in self.index_h(Z):
            cs.append(axes[0].contourf(xy, yx, U[n, :, :, k],
                                       zdir="z", offset=z, levels=self.LEVELS[id], cmap=self.CMAPS[id]))
        cbarlab = f'${id}(x, y, z)$'
        cbar1 = fig.colorbar(cs[0], ax=axes[0], shrink=0.9)
        cbar1.ax.set_ylabel(cbarlab, fontsize=14)

        plt.tight_layout()
        if self.save_fig:
            filename = self.name.joinpath(
                f'{self.TYPES[id]}_{coordlab}_t={t:.3g}.pdf')
            plt.savefig(filename)
        plt.show()

    def singlecross(self, U, t, n, x, d, axis, id):
        yz, zy = np.meshgrid(self.y, self.z, indexing="ij")
        xz, zx = np.meshgrid(self.x, self.z, indexing="ij")
        xy, yx = np.meshgrid(self.x, self.y, indexing="ij")

        fig = plt.figure()
        axes = []
        cs = []
        cbarlab = ""
        coordlab = ""

        axes.append(fig.add_subplot(1, 1, 1))
        plt.title(
            f'{self.TYPES[id]} distribution @ ${axis}={x:.3g}$ and $t =$ {t:.3g}')
        if axis == "x":
            cs = [axes[0].contourf(yz, zy, U[n, d, :, :],
                                   levels=self.LEVELS[id], cmap=self.CMAPS[id])]
            plt.xlabel(r"$y$", fontsize=16)
            plt.ylabel(r"$z$", fontsize=16)
            cbarlab = f'${id}(y, z)$'
        elif axis == "y":
            cs = [axes[0].contourf(xz, zx, U[n, :, d, :],
                                   levels=self.LEVELS[id], cmap=self.CMAPS[id])]
            plt.xlabel(r"$x$", fontsize=16)
            plt.ylabel(r"$z$", fontsize=16)
            cbarlab = f'${id}(x, z)$'
        elif axis == "z":
            cs = [axes[0].contourf(xy, yx, U[n, :, :, d],
                                   levels=self.LEVELS[id], cmap=self.CMAPS[id])]
            plt.xlabel(r"$x$", fontsize=16)
            plt.ylabel(r"$y$", fontsize=16)
            cbarlab = f'${id}(x, y)$'
        else:
            raise Exception()

        coordlab = f'{axis} = {d: .3g}'
        cbar1 = fig.colorbar(cs[0], ax=axes[0], shrink=0.9)
        cbar1.ax.set_ylabel(cbarlab, fontsize=14)

        plt.tight_layout()
        if self.save_fig:
            filename = self.name.joinpath(
                f'{self.TYPES[id]}__{coordlab}_t={t:.3g}.pdf')
            plt.savefig(filename)
        plt.show()

    def convert_to_array(self, A):
        if isinstance(A, np.ndarray):
            return A
        if isinstance(A, int) or isinstance(A, float):
            return np.array([A])
        if type(A) == list:
            return np.array(A)

    def index_t(self, T):
        T = self.convert_to_array(T)
        return zip(T, (T/(self.dt*self.t_jump)).round().astype(int))

    def index_h(self, X):
        X = self.convert_to_array(X)
        return zip(X, (X/self.dh).round().astype(int))

    def get_mode(self, X, Y, Z):
        if type(X) == int or type(X) == float:
            return "S", (X, round(X/self.dh), "x")
        if type(Y) == int or type(Y) == float:
            return "S", (Y, round(Y/self.dh), "y")
        if type(Z) == int or type(Z) == float:
            return "S", (Z, round(Z/self.dh), "z")
        return "M", None

    def set_latex(self, usetex):
        # Latex font rendering
        rc("font", **{"family": "serif", "serif": ["Palatino"]})
        rc("text", usetex=usetex)
