import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # For 3-d plott
from matplotlib import cm
import matplotlib.animation as animation

class Snapshot:
    def __init__(self, part):
        self.part = part
        self.filename = "untitled"
        self.save_fig = False

    def show_heat_map(self):
        plt.figure()
        #plt.plot(self.part.x, self.part.rhoArr, label=r"Prob density, $\rho$", color="b")
        plt.xlim(left=0, right = self.part.L)
        plt.ylim(bottom=0, top = 1)
        plt.legend()
        plt.grid()
        if self.save_fig:
            plt.savefig(self.filename + "_heatmap.pdf")
        plt.show()

    def show_boundary_cond(self):
        plt.figure()
        #plt.plot(self.part.x, self.part.reArrEven, label=r"$\Psi_R$", color="g", linewidth=0.75)
        #plt.plot(self.part.x, self.part.imArrEven, label=r"$\Psi_I$", color="m", linewidth=0.75)
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$y$", fontsize=16)
        plt.legend()
        plt.grid()
        if self.save_fig:
            plt.savefig(self.filename + "_BC.pdf")
        plt.show()

class Animation:
    def __init__(self, part, duration_irl, duration, fps):
        self.part = part
        self.duration_irl = duration_irl
        self.duration = duration
        self.fps = fps
        self.cut_n = int(duration/part.dt/duration_irl/fps)

        # Set up the figure and axes
        self.fig, self.ax1 = plt.subplots()

        # Initialize the line object
        self.line1, = self.ax1.plot([], [], lw=1.0, color='g', label=r"$\Psi_R$")
        self.line2, = self.ax1.plot([], [], lw=1.0, color='m', label=r"$\Psi_I$")
        self.line3, = self.ax1.plot([], [], lw=1.0, color='b', label=r"$|\Psi|^2$")
        self.line4, = self.ax1.plot([], [], lw=1.0, color='k', label=r"$V(x)$")
        self.lines = [self.line1, self.line2, self.line3]
        #   Plots potential (NOT in the same units at all)
        if self.part.hasPotential:
            self.line4.set_data(self.part.x, self.part.V/(1.5*self.part.E))
            self.lines.append(self.line4)

        # Set limits and labels for the axes
        self.ax1.set_xlim(left=0, right= part.L)
        self.ax1.set_ylim(bottom=-1, top=1)
        self.ax1.grid()

        # Actually do the animation
        self.anim = animation.FuncAnimation(self.fig, self.animate, repeat=False, frames=int(self.fps * self.duration_irl),
                                            interval=1000 / self.fps, blit=False)
        self.filename = "qm_particle_xs=" + str(self.part.xs) + "_sigma=" + str(self.part.sigmax) + "_pot=" + str(self.part.hasPotential) + ".mp4"

    def animate(self, i):
        print(i, "out of", self.fps * self.duration_irl)
        # Math that gets recalculated each iteration
        if (i != 0):
            for j in range(self.cut_n):
                self.part.calc_next()

        # Assigning the line object a set of values
        self.lines[0].set_data(self.part.x, self.part.reArrEven)
        self.lines[1].set_data(self.part.x, self.part.imArrEven)
        self.lines[2].set_data(self.part.x, self.part.rhoArr)

        # Uncomment the following line to save a hi-res version of each frame (mind the filenames though, they'll overwrite each other)
        # plt.savefig('test.png',format='png',dpi=600)

        return self.lines

    def run_no_threading(self):
        self.anim.save(self.filename, fps=self.fps, extra_args=['-vcodec', 'libx264'], dpi=200, bitrate=-1)