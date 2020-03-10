import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation

from Samplebeef import Beef

class Plotter:
    def __init__(self, beef, name = 'untitled', save_fig = False):
        """
        beef: A Beef object with data that one would want to plot.
        name: Filename for saved plots. Default: 'Untitled'
        save_fig: Determines whether the plots will be saved. Defauld: 'False'
        """
        self.U = beef.U
        self.C = beef.C
        
        self.t = beef.t
        self.x = beef.x
        self.y = beef.y
        self.z = beef.z

        self.dt = beef.dt
        self.h = beef.h

        self.filename = filename
        self.save_fig = save_fig
        

    def show_heat_map(self, t, x = None, y = None, z = None):
        if isinstance(t, list):
            pass
        else:
            n = int(t // self.dt)
            
            fig, ax = plt.subplots()
            plt.title(f'$t =$ {t:.2f}')
            cs = []
            cbarlab = ''
            
            if x is not None:
                i = int(x // self.h)
                yz, zy = np.meshgrid(self.y, self.z, indexing = 'ij')
                cs = [ax.contourf(yz, zy, self.U[n,i,:,:], cmap = cm.get_cmap('magma'))]
                plt.xlabel(r"$y$", fontsize=16)
                plt.ylabel(r"$z$", fontsize=16)
                cbarlab = r'$U(y,z)$'
            elif y is not None:
                j = int(y // self.h)
                xz, zx = np.meshgrid(self.x, self.z, indexing = 'ij')
                cs = [ax.contourf(xz, zx, self.U[n,:,j,:], 200, cmap = cm.get_cmap('magma'))]
                plt.xlabel(r"$x$", fontsize=16)
                plt.ylabel(r"$z$", fontsize=16)
                cbarlab = r'$U(x,z)$'
            elif z is not None:
                k = int(z // self.h)
                xy, yx = np.meshgrid(self.x, self.y, indexing = 'ij')
                cs = [ax.contourf(xy, yx, self.U[n,:,:,k], 200, cmap = cm.get_cmap('magma'))]
                plt.xlabel(r"$x$", fontsize=16)
                plt.ylabel(r"$y$", fontsize=16)
                cbarlab = r'$U(x,y)$'
            else:
                raise Exception("No crossection coordinate given.")
            
            cbar1 = fig.colorbar(cs[0], ax=ax, shrink=0.9)
            cbar1.ax.set_ylabel(cbarlab, fontsize=14)
            
            if self.save_fig:
                plt.savefig(self.name + f"_heatmap_t={t:.2f}.pdf")
            plt.show()

    def show_boundary_cond(self):
        plt.figure()
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$y$", fontsize=16)
        plt.legend()
        plt.grid()
        if self.save_fig:
            plt.savefig(self.filename + "_BC.pdf")
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

if (__name__ == '__main__'):
    h = 0.25
    dt = 0.1

    t = np.arange(0, 10.1, dt)
    x = np.arange(0, 5.25, h)
    y = np.arange(0, 4.25, h)
    z = np.arange(0, 3.25, h)

    d_shape = (len(t), len(x), len(y), len(z))
    U = np.random.random(d_shape)
    U[:,:,:, 0] = 0.00
    U[:,:, 0,:] = 0.25
    U[:,:,:, -1] = 0.50
    U[:,:, -1,:] = 0.75
    U[:, 0,:,:] = 1.00
    U[:, -1,:,:] = 1.25

    C = np.zeros(d_shape)

    bf = Beef(U, C, t, x, y, z)
    pl = Plotter(bf, save_fig=False)

    pl.show_heat_map(5, x = 2.50)
    pl.show_heat_map(5, y = 3.50)
    pl.show_heat_map(6, z = 2.0)

