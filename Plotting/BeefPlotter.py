import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
from pathlib import Path
from typing import Union
import json


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
            self.t = beefsim.t
            self.x = beefsim.x
            self.y = beefsim.y
            self.z = beefsim.z
            self.dt = beefsim.dt
            self.h = beefsim.dh

        self.name = name
        self.save_fig = save_fig

    def load_from_file(self, path: Path):
        if not isinstance(path, Path):
            path = Path(path)
        head_path = path.joinpath("header.json")
        temp_path = path.joinpath("T.dat")
        cons_path = path.joinpath("C.dat")

        header = None
        with open(head_path) as f:
            header = json.load(f)

        self.dt = header["dt"]
        self.h = header["dh"]

        dims = header["dims"]
        shape = header["shape"]
        self.t = np.linspace(header["t0"], header["tn"], shape[0])
        self.x = np.linspace(dims["x0"], dims["xn"], shape[1])
        self.y = np.linspace(dims["y0"], dims["yn"], shape[2])
        self.z = np.linspace(dims["z0"], dims["zn"], shape[3])

        self.T_data = np.memmap(temp_path,
                                dtype="float64",
                                mode="r",
                                shape=tuple(shape))
        self.C_data = np.memmap(cons_path,
                                dtype="float64",
                                mode="r",
                                shape=tuple(shape))

    def show_heat_map2(self, t, id, x=None, y=None, z=None):
        U_data = self.T_data if id == "T" else self.C_data
        self.show_heat_map(U_data, t, id, x, y, z)

    def show_heat_map(self, U_data, t, id, x=None, y=None, z=None):
        if id == 'T':
            self.__shm_temp(U_data, t, x, y, z)
        elif id == 'C':
            self.__shm_cons(U_data, t, x, y, z)
        else:
            raise ValueError(
                'Trying to aquire a quantity that does not exist.')

    def __shm_temp(self, U_data, t, x=None, y=None, z=None):
        T = U_data

        if isinstance(t, list):
            pass
        else:
            n = int(t // self.dt)

            fig, ax = plt.subplots()
            cs = []
            cbarlab = ''
            coordlab = ''

            if x is not None:
                if isinstance(x, list):
                    pass
                else:
                    plt.title(f'X-section @ $x={x:.3g}$ and $t =$ {t:.3g}')
                    i = int(x // self.h)
                    yz, zy = np.meshgrid(self.y, self.z, indexing='ij')
                    cs = [ax.contourf(yz, zy, T[n, i, :, :], 65,
                                      cmap=cm.get_cmap('magma'))]
                    plt.xlabel(r"$y$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$T(y,z)$'
                    coordlab = f'x={x:.3g}'
            elif y is not None:
                if isinstance(y, list):
                    pass
                else:
                    plt.title(f'X-section @ $y={y:.3g}$ and $t =$ {t:.3g}')
                    j = int(y // self.h)
                    xz, zx = np.meshgrid(self.x, self.z, indexing='ij')
                    cs = [ax.contourf(xz, zx, T[n, :, j, :], 65,
                                      cmap=cm.get_cmap('magma'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$T(x,z)$'
                    coordlab = f'y={y:.3g}'
            elif z is not None:
                if isinstance(z, list):
                    pass
                else:
                    plt.title(f'X-section @ $z={z:.3g}$ and $t =$ {t:.3g}')
                    k = int(z // self.h)
                    xy, yx = np.meshgrid(self.x, self.y, indexing='ij')
                    cs = [ax.contourf(xy, yx, T[n, :, :, k], 65,
                                      cmap=cm.get_cmap('magma'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$y$", fontsize=16)
                    cbarlab = r'$T(x,y)$'
                    coordlab = f'z={z:.3g}'
            else:
                raise Exception("No crossection coordinate given.")

            cbar1 = fig.colorbar(cs[0], ax=ax, shrink=0.9)
            cbar1.ax.set_ylabel(cbarlab, fontsize=14)

            if self.save_fig:
                filename = self.name.joinpath(
                    f'heatmap_{coordlab}_t={t:.3g}.png')
                plt.savefig(filename)
            plt.show()

    def __shm_cons(self, U_data, t, x=None, y=None, z=None):
        C = U_data

        if isinstance(t, list):
            pass
        else:
            n = int(t // self.dt)

            fig, ax = plt.subplots()
            cs = []
            cbarlab = ''
            coordlab = ''

            if x is not None:
                if isinstance(x, list):
                    pass
                else:
                    plt.title(f'X-section @ $x={x:.3g}$ and $t =$ {t:.3g}')
                    i = int(x // self.h)
                    yz, zy = np.meshgrid(self.y, self.z, indexing='ij')
                    cs = [ax.contourf(yz, zy, C[n, i, :, :], 65,
                                      cmap=cm.get_cmap('viridis'))]
                    plt.xlabel(r"$y$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$C(y,z)$'
                    coordlab = f'x={x:.3g}'
            elif y is not None:
                if isinstance(y, list):
                    pass
                else:
                    plt.title(f'X-section @ $y={y:.3g}$ and $t =$ {t:.3g}')
                    j = int(y // self.h)
                    xz, zx = np.meshgrid(self.x, self.z, indexing='ij')
                    cs = [ax.contourf(xz, zx, C[n, :, j, :], 65,
                                      cmap=cm.get_cmap('viridis'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$C(x,z)$'
                    coordlab = f'y={y:.3g}'
            elif z is not None:
                if isinstance(z, list):
                    pass
                else:
                    plt.title(f'X-section @ $z={z:.3g}$ and $t =$ {t:.3g}')
                    k = int(z // self.h)
                    xy, yx = np.meshgrid(self.x, self.y, indexing='ij')
                    cs = [ax.contourf(xy, yx, C[n, :, :, k], 65,
                                      cmap=cm.get_cmap('viridis'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$y$", fontsize=16)
                    cbarlab = r'$C(x,y)$'
                    coordlab = f'z={z:.3g}'
            else:
                raise Exception("No crossection coordinate given.")

            cbar1 = fig.colorbar(cs[0], ax=ax, shrink=0.9)
            cbar1.ax.set_ylabel(cbarlab, fontsize=14)

            if self.save_fig:
                filename = self.name.joinpath(
                    f'heatmap_{coordlab}_t={t:.3g}.png')
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
