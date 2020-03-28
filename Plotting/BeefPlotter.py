import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation


class Plotter:
    def __init__(self, beefsim, name='untitled', save_fig=False):
        """
        beefsim: A BeefSimulator object with axis and stepping data for plotting.
        name: Filename for saved plots. Default: 'untitled'
        save_fig: Determines whether the plots will be saved. Default: 'False'
        """
        self.t = beefsim.t
        self.x = beefsim.x
        self.y = beefsim.y
        self.z = beefsim.z

        self.dt = beefsim.dt
        self.h = beefsim.dh

        self.name = name
        self.save_fig = save_fig

    def show_heat_map(self, U_data, t, x=None, y=None, z=None):
        U = U_data
        print(U.shape)
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
                    cs = [ax.contourf(yz, zy, U[n, i, :, :], 65,
                                    cmap=cm.get_cmap('magma'))]
                    plt.xlabel(r"$y$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$U(y,z)$'
                    coordlab = f'x={x:.3g}'
            elif y is not None:
                if isinstance(y, list):
                    pass
                else:
                    plt.title(f'X-section @ $y={y:.3g}$ and $t =$ {t:.3g}')
                    j = int(y // self.h)
                    xz, zx = np.meshgrid(self.x, self.z, indexing='ij')
                    cs = [ax.contourf(xz, zx, U[n, :, j, :], 65,
                                    cmap=cm.get_cmap('magma'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$z$", fontsize=16)
                    cbarlab = r'$U(x,z)$'
                    coordlab = f'y={y:.3g}'
            elif z is not None:
                if isinstance(z, list):
                    pass
                else:
                    plt.title(f'X-section @ $z={z:.3g}$ and $t =$ {t:.3g}')
                    k = int(z // self.h)
                    xy, yx = np.meshgrid(self.x, self.y, indexing='ij')
                    cs = [ax.contourf(xy, yx, U[n, :, :, k], 65,
                                    cmap=cm.get_cmap('magma'))]
                    plt.xlabel(r"$x$", fontsize=16)
                    plt.ylabel(r"$y$", fontsize=16)
                    cbarlab = r'$U(x,y)$'
                    coordlab = f'z={z:.3g}'
            else:
                raise Exception("No crossection coordinate given.")

            cbar1 = fig.colorbar(cs[0], ax=ax, shrink=0.9)
            cbar1.ax.set_ylabel(cbarlab, fontsize=14)

            if self.save_fig:
                plt.savefig(self.name + "_heatmap_" + coordlab + f"_t={t:.3g}.png")
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
        #self.ax1.set_xlim(left=0, right=)
        #self.ax1.set_ylim(bottom=-1, top=1)
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
