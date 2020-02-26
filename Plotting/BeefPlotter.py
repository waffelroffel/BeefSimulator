import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # For 3-d plott
from matplotlib import cm
import matplotlib.animation as animation

class Plotter:
    def __init__(self, beef, filename = 'untitled', save_fig = False):
        self.beef = beef
        self.filename = filename
        self.save_fig = save_fig
        self.n = beef.t / self.beef.dt

    def show_heat_map(self, t):
        fig, ax = plt.subplots()

        cs1 = [ax.contourf(self.xv, self.yv, self.beef.U[:,:,self.n], 65, cmap = cm.magma)]
        cbar1 = fig.colorbar(cs1[0], ax=ax, shrink=0.9)
        cbar1.ax.set_ylabel(r'$U(x,y)$', fontsize=14)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$y$", fontsize=16)
        plt.tight_layout()
        plt.legend()
        if self.save_fig:
            plt.savefig(self.filename + "_heatmap.pdf")
        plt.show()

    def show_boundary_cond(self):
        plt.figure()
        #plt.plot(self.system.x, self.system.reArrEven, label=r"$\Psi_R$", color="g", linewidth=0.75)
        #plt.plot(self.system.x, self.system.imArrEven, label=r"$\Psi_I$", color="m", linewidth=0.75)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$y$", fontsize=16)
        plt.legend()
        plt.grid()
        if self.save_fig:
            plt.savefig(self.filename + "_BC.pdf")
        plt.show()


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