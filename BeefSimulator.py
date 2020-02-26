import Plotting.BeefPlotter as BP
import numpy as np
import matplotlib.pyplot as plt
from Beef import Beef


class BeefSimulator:
    def __init__(self, dims, h, alpha, conds=None, plotter=BP.Plotter):
        """
        dims: [ [x_start, x_len], [y_start, y_len], ... , [t_start, t_len] ]
        h: step in each dim
        conds: [t_0(x,y,z,...), x_0(t), x_1(t), y_0(t), y_1(t), ...]
        initial condition and boundry conditions in that order
        """

        self.plotter = plotter
		self.beef = Beef(dims, h)

        assert len(dims) != 0, "No dimensions found"

        self.D = len(dims)-1

        steps = [(s, s+l, int((l-s)/h)) for s, l in dims[:-1]]

        self.space = [np.linspace(*step) for step in steps]

        self.h = h
        dt = h**2/(4*alpha)
        print(f'{dt=}')
        t_start = dims[-1][0]
        t_len = dims[-1][1]
        t_end = t_start+t_len
        self.t_steps = int((t_len)/dt)+1
        self.time = np.linspace(t_start, t_end, self.t_steps)

        self.mu = dt * alpha / h**2
        print(f'steps = {*[step[2] for step in steps], self.t_steps}')
        self.U = np.zeros([step[2] for step in steps]+[self.t_steps])

        self.t_n = t_start

        self.conds = conds

    def apply_conditions(self, conds=None):
        if conds == None and self.conds == None:
            raise Exception("no conditions given")
        if conds == None:
            conds = self.conds
        mesh = np.meshgrid(*reversed(self.space))  # fix
        if self.D == 3:
            pass
        elif self.D == 2:
            print(self.U[..., 0].shape)
            self.U[0, ...] = conds[1](self.time)
            self.U[-1, :, :] = conds[2](self.time)
            self.U[:, 0, :] = conds[3](self.time)
            self.U[:, -1, :] = conds[4](self.time)
            self.U[..., 0] = conds[0](*mesh)
        elif self.D == 1:
            pass

    def _neuman_omega(self, x=None, y=None, z=None, method="cd"):
        if method == "cd":
            if x:
                x = -1 if x == 1 else 0
                self.U[x, 1:-1, self.t_n+1] = self.U[x, 1:-1, self.t_n] \
                    + self.mu*(self.U[x+1, 1:-1, self.t_n]
                               - 2*self.h*self.conds[abs(x)+1](self.space[0])
                               - 4*self.U[x, 1:-1, self.t_n]
                               + self.U[x, :-2, self.t_n]
                               + self.U[x, 2:, self.t_n])
            if y:
                pass
            if z:
                pass

    def solve_next(self, method="cd"):
        if self.t_n == self.t_steps-1:
            return False
        if method == "cd":
            # self._neuman_omega(1)
            self.U[1:-1, 1:-1, self.t_n + 1] = self.U[1:-1, 1:-1, self.t_n] \
                + self.mu*(self.U[1:-1, :-2, self.t_n]
                           + self.U[:-2, 1:-1, self.t_n]
                           - 4*self.U[1:-1, 1:-1, self.t_n]
                           + self.U[2:, 1:-1, self.t_n]
                           + self.U[1:-1, 2:, self.t_n])
        if self.t_n % 100 == 0:
            print(f'{self.t_n=}')
        self.t_n += 1
        return True

    def solve_all(self, method="cd"):
        print("iterating")
        while self.solve_next(method):
            pass
        print("done")

    def plot(self, steps=20):
        for i in range(self.U.shape[-1]):  # pylint: disable=E1136
            if i % steps != 0:
                continue
            print(f'{i=}')
            plt.imshow(self.U[..., i])
            plt.show()
