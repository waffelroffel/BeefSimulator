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
        self.vmin_T, self.vmax_T = 12, 14
        # np.min(beefsim.C0), np.max(beefsim.C0)
        self.vmin_C, self.vmax_C = 10, 50

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
    """
    def __shm_temp(self, U_data, t, x=None, y=None, z=None):
        T = U_data

        if isinstance(t, list):
            # TODO: Implement this
            pass
        else:
            # n = int(t / self.t_jump)
            n = 0 if (self.t_jump == -1) else \
                int(round(t / (self.dt * self.t_jump)))

            fig = plt.figure()
            axes = []
            cs = []
            cbarlab = ""
            coordlab = ""

            if x is None:
                yz, zy = np.meshgrid(self.y, self.z, indexing="ij")
                if isinstance(x, list):
                    init_3d(fig, axes)
                    axes[0].text2D(
                        0.5, 0.95, f'Temperature distribution @ $t =$ {t:.3g}', transform=axes[0].transAxes)
                    axes[0].set_xlim3d(x[0], x[len(x) - 1])
                    axes[0].view_init(15, - 107)
                    for x_ in x:
                        i = int(round((x_ / self.dh)))
                        cs.append(axes[0].contourf(T[n, i, :, :], yz, zy, zdir="x", offset=x_, levels=self.levels_T,
                                                   cmap=cm.get_cmap("magma")))
                    cbarlab = r"$T(x,y,z)$"

                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(
                        f'Temperature distribution @ $x={x:.3g}$ and $t =$ {t:.3g}')
                    i = int(round((x_ / self.dh)))
                    cs = [axes[0].contourf(yz, zy, T[n, i, :, :], levels=self.levels_T,
                                           cmap=cm.get_cmap("magma"))]
                    plt.xlabel(r"$y$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r"$T(y,z)$"
                    coordlab = f'x={x:.3g}'
            elif y is not None:
                xz, zx = np.meshgrid(self.x, self.z, indexing="ij")
                if isinstance(y, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f'Temperature distribution @ $t =$ {t:.3g}',
                                   transform=axes[0].transAxes)
                    axes[0].set_ylim3d(y[0], y[len(y) - 1])
                    axes[0].view_init(15, - 16)
                    for y_ in y:
                        j = int(round(y_ / self.dh))
                        cs.append(axes[0].contourf(xz, T[n, :, j, :], zx,  offset=y_, zdir="y", levels=self.levels_T,
                                                   cmap=cm.get_cmap("magma")))
                    cbarlab = r"$T(x,y,z)$"
                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(
                        f'Temperature distribution @ $y={y:.3g}$ and $t =$ {t:.3g}')
                    j = int(y // self.dh)
                    cs = [axes[0].contourf(xz, zx, T[n, :, j, :], levels=self.LEVELS["T"],
                                           cmap=cm.get_cmap("magma"))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r"$T(x,z)$"
                    coordlab = f'y={y:.3g}'
            elif z is not None:
                xy, yx = np.meshgrid(self.x, self.y, indexing="ij")
                if isinstance(z, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f'Temperature distribution @ $t =$ {t:.3g}',
                                   transform=axes[0].transAxes)
                    axes[0].set_zlim3d(z[0], z[len(z) - 1])
                    axes[0].view_init(12, - 30)
                    for z_ in z:
                        k = int(round((z_ / self.dh)))
                        cs.append(axes[0].contourf(xy, yx, T[n, :, :, k], zdir="z", offset=z_, levels=self.levels_T,
                                                   cmap=cm.get_cmap("magma")))
                    cbarlab = r"$T(x,y,z)$"
                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(
                        f'Temperature distribution @ $z={z:.3g}$ and $t =$ {t:.3g}')
                    k = int(z // self.dh)
                    cs = [axes[0].contourf(xy, yx, T[n, :, :, k], levels=self.levels_T,
                                           cmap=cm.get_cmap("magma"))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$y$", fontsize=16)
                    cbarlab = r"$T(x,y)$"
                    coordlab = f'z={z:.3g}'
            else:
                raise Exception("No crossection coordinate given.")

            cbar1 = fig.colorbar(cs[0], ax=axes[0], shrink=0.9)
            cbar1.ax.set_ylabel(cbarlab, fontsize=14)

            plt.tight_layout()
            if self.save_fig:
                filename = self.name.joinpath(
                    f'tempmap_{coordlab}_t={t:.3g}.pdf')
                plt.savefig(filename)
            plt.show()

    def __shm_cons(self, U_data, t, x=None, y=None, z=None):
        C = U_data

        if isinstance(t, list):
            # TODO: Implement this
            pass
        else:
            n = int(t // self.t_jump)

            fig = plt.figure()
            axes = []
            cs = []
            cbarlab = ""
            coordlab = ""

            if x is not None:
                yz, zy = np.meshgrid(self.y, self.z, indexing="ij")
                if isinstance(x, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f'Concentration distribution @ $t =$ {t:.3g}',
                                   transform=axes[0].transAxes)
                    axes[0].set_xlim3d(x[0], x[len(x) - 1])
                    axes[0].view_init(15, - 107)
                    for x_ in x:
                        i = int(round((x_ / self.dh)))
                        cs.append(axes[0].contourf(C[n, i, :, :], yz, zy, zdir="x", offset=x_, levels=self.levels_C,
                                                   cmap=cm.get_cmap("viridis")))
                    cbarlab = r"$C(x,y,z)$"
                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(
                        f'Concentration distribution @ $x={x:.3g}$ and $t =$ {t:.3g}')
                    i = int(round((x_ / self.dh)))
                    cs = [axes[0].contourf(yz, zy, C[n, i, :, :], levels=self.levels_C,
                                           cmap=cm.get_cmap("viridis"))]
                    plt.xlabel(r"$y$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r"$C(y,z)$"
                    coordlab = f'x={x:.3g}'
            elif y is not None:
                xz, zx = np.meshgrid(self.x, self.z, indexing="ij")
                if isinstance(y, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f'Concentration distribution @ $t =$ {t:.3g}',
                                   transform=axes[0].transAxes)
                    axes[0].set_ylim3d(y[0], y[len(y) - 1])
                    axes[0].view_init(15, - 16)
                    for y_ in y:
                        j = int(round(y_ / self.dh))
                        cs.append(axes[0].contourf(xz, C[n, :, j, :], zx,  offset=y_, zdir="y", levels=self.levels_C,
                                                   cmap=cm.get_cmap("viridis")))
                    cbarlab = r"$C(x,y,z)$"
                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(
                        f'Concentration distribution @ $y={y:.3g}$ and $t =$ {t:.3g}')
                    j = int(y // self.dh)
                    cs = [axes[0].contourf(xz, zx, C[n, :, j, :], levels=self.levels_C,
                                           cmap=cm.get_cmap("viridis"))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r"$C(x,z)$"
                    coordlab = f'y={y:.3g}'
            elif z is not None:
                xy, yx = np.meshgrid(self.x, self.y, indexing="ij")
                if isinstance(z, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f'Concentration distribution @ $t =$ {t:.3g}',
                                   transform=axes[0].transAxes)
                    axes[0].set_zlim3d(z[0], z[len(z) - 1])
                    axes[0].view_init(12, - 30)
                    for z_ in z:
                        k = int(round((z_ / self.dh)))
                        cs.append(axes[0].contourf(xy, yx, C[n, :, :, k], zdir="z", offset=z_, levels=self.levels_C,
                                                   cmap=cm.get_cmap("viridis")))
                    cbarlab = r"$C(x,y,z)$"
                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(
                        f'Concentration distribution @ $z={z:.3g}$ and $t =$ {t:.3g}')
                    k = int(z // self.dh)
                    cs = [axes[0].contourf(xy, yx, C[n, :, :, k], levels=self.levels_C,
                                           cmap=cm.get_cmap("viridis"))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$y$", fontsize=16)
                    cbarlab = r"$C(x,y)$"
                    coordlab = f'z={z:.3g}'
            else:
                raise Exception("No crossection coordinate given.")

            cbar1 = fig.colorbar(cs[0], ax=axes[0], shrink=0.9)
            cbar1.ax.set_ylabel(cbarlab, fontsize=14)

            plt.tight_layout()
            if self.save_fig:
                filename = self.name.joinpath(
                    f'consmap_{coordlab}_t={t:.3g}.pdf')
                plt.savefig(filename)
            plt.show()
    """

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
