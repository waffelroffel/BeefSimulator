import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
from pathlib import Path
from typing import Union
import json


def init_3d(fig, axes):
    axes.append(fig.gca(projection='3d'))
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$y$')
    axes[0].set_zlabel('$z$')


class Plotter:
    def __init__(self, beefsim=None, name='untitled', save_fig=False):
        """
        beefsim: A BeefSimulator object with axis and stepping data for plotting.
        name: Filename for saved plots. Default: 'untitled'
        save_fig: Determines whether the plots will be saved. Default: 'False'
        """
        if beefsim is None:
            self.load_from_file(name)
        else:
            self.load_from_class(beefsim)

        self.levels_T = np.linspace(self.vmin_T, self.vmax_T, 65)
        self.levels_C = np.linspace(self.vmin_C, self.vmax_C, 65)
        self.name = name
        self.save_fig = save_fig

    def load_from_class(self, beefsim):
        self.t = beefsim.t
        self.x = beefsim.x
        self.y = beefsim.y
        self.z = beefsim.z
        self.t_jump = beefsim.t_jump
        self.dt = beefsim.dt
        self.dh = beefsim.dh
        # TODO: not garanteed to get min max from first timestep
        # np.min(beefsim.T0), np.max(beefsim.T0)
        self.vmin_T, self.vmax_T = 12.9999, 13.0001
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
        self.vmin_T = np.min(self.T_data[0])
        self.vmax_T = np.max(self.T_data[0])
        self.vmin_C = np.min(self.C_data[0])
        self.vmax_C = np.max(self.C_data[0])

    def show_heat_map2(self, t, id, x=None, y=None, z=None):
        U_data = self.T_data if id == "T" else self.C_data
        self.show_heat_map(U_data, t, id, x, y, z)

    def show_heat_map(self, U_data, t, id, x=None, y=None, z=None, multi=False):
        if id == 'T':
            if multi:
                self.multicross(U_data, t, x, y, z, self.levels_T)
            else:
                self.__shm_temp(U_data, t, x, y, z)
        elif id == 'C':
            if multi:
                self.multicross(U_data, t, x, y, z, self.levels_C)
            else:
                self.__shm_cons(U_data, t, x, y, z)
        else:
            raise ValueError(
                'Trying to aquire a quantity that does not exist.')

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
            cbarlab = ''
            coordlab = ''

            if x is not None:
                yz, zy = np.meshgrid(self.y, self.z, indexing='ij')
                if isinstance(x, list):
                    init_3d(fig, axes)
                    axes[0].text2D(
                        0.5, 0.95, f"Temperature @ $t =$ {t:.3g}", transform=axes[0].transAxes)
                    axes[0].set_xlim3d(x[0], x[len(x) - 1])
                    for x_ in x:
                        i = int(round((x_ / self.dh)))
                        cs.append(axes[0].contourf(T[n, i, :, :], yz, zy, zdir='x', offset=x_, levels=self.levels_T,
                                                   cmap=cm.get_cmap('magma')))
                    cbarlab = r'$T(x,y,z)$'

                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(f'Temperature @ $x={x:.3g}$ and $t =$ {t:.3g}')
                    i = int(round((x_ / self.dh)))
                    cs = [axes[0].contourf(yz, zy, T[n, i, :, :], levels=self.levels_T,
                                           cmap=cm.get_cmap('magma'))]
                    plt.xlabel(r"$y$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$T(y,z)$'
                    coordlab = f'x={x:.3g}'
            elif y is not None:
                xz, zx = np.meshgrid(self.x, self.z, indexing='ij')
                if isinstance(y, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f"Temperature @ $t =$ {t:.3g}",
                                   transform=axes[0].transAxes)
                    axes[0].set_ylim3d(y[0], y[len(y) - 1])
                    for y_ in y:
                        j = int(round(y_ / self.dh))
                        cs.append(axes[0].contourf(xz, T[n, :, j, :], zx,  offset=y_, zdir='y', levels=self.levels_T,
                                                   cmap=cm.get_cmap('magma')))
                    cbarlab = r'$T(x,y,z)$'
                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(f'Temperature @ $y={y:.3g}$ and $t =$ {t:.3g}')
                    j = int(y // self.dh)
                    cs = [axes[0].contourf(xz, zx, T[n, :, j, :], levels=self.levels_T,
                                           cmap=cm.get_cmap('magma'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$T(x,z)$'
                    coordlab = f'y={y:.3g}'
            elif z is not None:
                xy, yx = np.meshgrid(self.x, self.y, indexing='ij')
                if isinstance(z, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f"Temperature @ $t =$ {t:.3g}",
                                   transform=axes[0].transAxes)
                    axes[0].set_zlim3d(z[0], z[len(z) - 1])
                    for z_ in z:
                        k = int(round((z_ / self.dh)))
                        cs.append(axes[0].contourf(xy, yx, T[n, :, :, k], zdir='z', offset=z_, levels=self.levels_T,
                                                   cmap=cm.get_cmap('magma')))
                    cbarlab = r'$T(x,y,z)$'
                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(f'Temperature @ $z={z:.3g}$ and $t =$ {t:.3g}')
                    k = int(z // self.dh)
                    cs = [axes[0].contourf(xy, yx, T[n, :, :, k], levels=self.levels_T,
                                           cmap=cm.get_cmap('magma'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$y$", fontsize=16)
                    cbarlab = r'$T(x,y)$'
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
            cbarlab = ''
            coordlab = ''

            if x is not None:
                yz, zy = np.meshgrid(self.y, self.z, indexing='ij')
                if isinstance(x, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f"Concentration @ $t =$ {t:.3g}",
                                   transform=axes[0].transAxes)
                    axes[0].set_xlim3d(x[0], x[len(x) - 1])
                    for x_ in x:
                        i = int(round((x_ / self.dh)))
                        cs.append(axes[0].contourf(C[n, i, :, :], yz, zy, zdir='x', offset=x_, levels=self.levels_C,
                                                   cmap=cm.get_cmap('viridis')))
                    cbarlab = r'$C(x,y,z)$'

                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(f'Concentration @ $x={x:.3g}$ and $t =$ {t:.3g}')
                    i = int(round((x_ / self.dh)))
                    cs = [axes[0].contourf(yz, zy, C[n, i, :, :], levels=self.levels_C,
                                           cmap=cm.get_cmap('viridis'))]
                    plt.xlabel(r"$y$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$C(y,z)$'
                    coordlab = f'x={x:.3g}'
            elif y is not None:
                xz, zx = np.meshgrid(self.x, self.z, indexing='ij')
                if isinstance(y, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f"Concentration @ $t =$ {t:.3g}",
                                   transform=axes[0].transAxes)
                    axes[0].set_ylim3d(y[0], y[len(y) - 1])
                    for y_ in y:
                        j = int(round(y_ / self.dh))
                        cs.append(axes[0].contourf(xz, C[n, :, j, :], zx,  offset=y_, zdir='y', levels=self.levels_C,
                                                   cmap=cm.get_cmap('viridis')))
                    cbarlab = r'$C(x,y,z)$'
                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(f'Concentration @ $y={y:.3g}$ and $t =$ {t:.3g}')
                    j = int(y // self.dh)
                    cs = [axes[0].contourf(xz, zx, C[n, :, j, :], levels=self.levels_C,
                                           cmap=cm.get_cmap('viridis'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$C(x,z)$'
                    coordlab = f'y={y:.3g}'
            elif z is not None:
                xy, yx = np.meshgrid(self.x, self.y, indexing='ij')
                if isinstance(z, list):
                    init_3d(fig, axes)
                    axes[0].text2D(0.5, 0.95, f"Concentration @ $t =$ {t:.3g}",
                                   transform=axes[0].transAxes)
                    axes[0].set_zlim3d(z[0], z[len(z) - 1])
                    for z_ in z:
                        k = int(round((z_ / self.dh)))
                        cs.append(axes[0].contourf(xy, yx, C[n, :, :, k], zdir='z', offset=z_, levels=self.levels_C,
                                                   cmap=cm.get_cmap('viridis')))
                    cbarlab = r'$C(x,y,z)$'
                else:
                    axes.append(fig.add_subplot(1, 1, 1))
                    plt.title(f'Concentration @ $z={z:.3g}$ and $t =$ {t:.3g}')
                    k = int(z // self.dh)
                    cs = [axes[0].contourf(xy, yx, C[n, :, :, k], levels=self.levels_C,
                                           cmap=cm.get_cmap('viridis'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$y$", fontsize=16)
                    cbarlab = r'$C(x,y)$'
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

    def show_boundary_cond(self):
        plt.figure()
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$y$", fontsize=16)
        plt.legend()
        plt.grid()
        if self.save_fig:
            plt.savefig(self.name + "_BC.png")
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

    def multicross(self, U, T, X, Y, Z, levels):
        for t, n in self.index_t(T):
            self._multicross(U, t, n, X, Y, Z, levels)

    def _multicross(self, U, t, n, X, Y, Z, levels):
        # TODO: move outside
        # change cmap color pallett
        yz, zy = np.meshgrid(self.y, self.z, indexing='ij')
        xz, zx = np.meshgrid(self.x, self.z, indexing='ij')
        xy, yx = np.meshgrid(self.x, self.y, indexing='ij')

        fig = plt.figure()
        axes = []
        cs = []
        cbarlab = ''
        coordlab = ''

        init_3d(fig, axes)
        axes[0].text2D(
            0.5, 0.95, f"Temperature @ $t =$ {t:.3g}", transform=axes[0].transAxes)

        axes[0].set_xlim3d(self.x[0], self.x[-1])
        axes[0].set_ylim3d(self.y[0], self.y[-1])
        axes[0].set_zlim3d(self.z[0], self.z[-1])

        for x, i in self.index_h(X):
            cs.append(axes[0].contourf(U[n, i, :, :], yz, zy,
                                       levels=levels, zdir='x', offset=x, cmap=cm.get_cmap('RdBu')))
        for y, j in self.index_h(Y):
            cs.append(axes[0].contourf(xz, U[n, :, j, :], zx,
                                       levels=levels, zdir='y', offset=y, cmap=cm.get_cmap('RdBu')))
        for z, k in self.index_h(Z):
            cs.append(axes[0].contourf(xy, yx, U[n, :, :, k],
                                       levels=levels, zdir='z', offset=z, cmap=cm.get_cmap('RdBu')))
        cbarlab = r'$T(x,y,z)$'
        cbar1 = fig.colorbar(cs[0], ax=axes[0], shrink=0.9)
        cbar1.ax.set_ylabel(cbarlab, fontsize=14)

        plt.tight_layout()
        if self.save_fig:
            filename = self.name.joinpath(
                f'tempmap_{coordlab}_t={t:.3g}.pdf')
            plt.savefig(filename)
        plt.show()


'''
class Animation:
    def __init__(self, system, duration_irl, duration, fps):
        self.system = system
        self.duration_irl = duration_irl
        self.duration = duration
        self.fps = fps
        self.cut_n = int(duration/system.dt/duration_irl/fps)

        # Set up the figure and axes
        self.fig, self.ax1 = plt.subplots()

        # Initialize the line object
        self.line, = self.ax1.plot(
            [], [], lw=1.0, color='g', label=r"")

        # Set limits and labels for the axes
        # self.ax1.set_xlim(left=0, right=)
        # self.ax1.set_ylim(bottom=-1, top=1)
        self.ax1.grid()

        # Actually do the animation
        self.anim = animation.FuncAnimation(self.fig, self.animate, repeat=False, frames=int(self.fps * self.duration_irl),
                                            interval=1000 / self.fps, blit=False)
        self.filename = "default.mp4"

    def animate(self, i):
        print(i, "out of", self.fps * self.duration_irl)
        # Math that gets recalculated each iteration
        if (i != 0):
            for j in range(self.cut_n):
                self.system.calc_next()

        # Assigning the line object a set of values
        self.lines[0].set_data(self.system.x, self.system.reArrEven)


        # Uncomment the following line to save a hi-res version of each frame (mind the filenames though, they'll overwrite each other)
        # plt.savefig('test.png',format='png',dpi=600)

        return self.lines

    def run_no_threading(self):
        self.anim.save(self.filename, fps=self.fps, extra_args=[
                       '-vcodec', 'libx264'], dpi=200, bitrate=-1)
'''
