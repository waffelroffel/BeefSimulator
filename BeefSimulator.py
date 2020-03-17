import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import Plotting.BeefPlotter as BP


class BeefSimulator:
    def __init__(self, dims, a, b, c, alpha, beta, gamma, initial, dh=0.01, dt=0.1, filename="data.csv", logging=1):
        """
        dims: [ [x_start, x_len], [y_start, y_len], ... , [t_start, t_len] ]

        pde: a*dT_dt = b*T_nabla + c*T_gradient

        boundary: alpha*T_gradient + beta*T = gamma

        dh: stepsize in each dim

        dt: time step

        initial: scalar or function with parameter (x,y,z,t)

        logging:
         - 0: nothing
         - 1: only initial setup and end state
         - 2: time steps
         - 3: A and b
         - 4: everything
        """
        """
        TODO:
        - [X] change 3D cordinates to 1D indexing: T1 = T0 + ( A @ T0 + b )
        - [X] contruct A and b
        - [ ] make it work with only direchet boundary: alpha = 0
        - [ ] change a, b, c, alpha, beta, gamma to functions
        - [ ] validate with manufactored solution
        - [ ] implement C (concentration)
        - [ ] T and C coupled
        - [ ] add plotter
        - [ ] add data management
        - [ ] add another logg level between 1 and 2 for linspaces and initial state
        """

        # Defines the PDEs and boundary conditions
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.dh = dh
        self.dt = dt

        xyz_steps = [(s, s+l, int(l/dh+1)) for s, l in dims[:-1]]
        t_steps = (dims[-1][0], dims[-1][0]+dims[-1][1], int(dims[-1][1]/dt+1))

        self.x = np.linspace(*xyz_steps[0])
        self.y = np.linspace(*xyz_steps[1])
        self.z = np.linspace(*xyz_steps[2])
        self.t = np.linspace(*t_steps)

        self.shape = (self.x.size, self.y.size, self.z.size)
        self.I, self.J, self.K = self.shape
        self.n = self.I * self.J * self.K
        self.inner = (self.I-2) * (self.J-2) * (self.K-2)
        self.border = self.n-self.inner

        # rename: the 1D indicies for all the boundary points
        self.bis = self.find_border_indicies()

        # send it through self.index_of before use
        xx, yy, zz = np.meshgrid(self.x, self.y, self.z)

        self.T1 = np.zeros(self.n)
        self.T0 = np.zeros(self.n)
        self.T0[...] = initial(xx, yy, zz) if callable(
            initial) else initial  # currently does't support function

        self.filename = filename
        self.save([])  # save header data: dims, time steps, ...
        self.save(self.T0)

        self.plotter = BP.Plotter(self)

        self.logging = logging

        self.logg(1, f'SETUP FINISHED. LOGGING...')
        self.logg(1, f'----------------------------------------------')
        self.logg(1, f'Logging level:       {self.logging}')
        self.logg(1, f'Shape:               {self.shape}')
        self.logg(1, f'Total nodes:         {self.n}')
        self.logg(1, f'Inner nodes:         {self.inner}')
        self.logg(1, f'Boundary nodes:      {self.border}')
        self.logg(
            1, f'x linspace:          dx: {self.dh}, \t x: {self.x[0]} -> {self.x[-1]}, \t steps: {self.x.size}')
        self.logg(
            1, f'y linspace:          dy: {self.dh}, \t y: {self.y[0]} -> {self.y[-1]}, \t steps: {self.y.size}')
        self.logg(
            1, f'z linspace:          dz: {self.dh}, \t z: {self.z[0]} -> {self.z[-1]}, \t steps: {self.z.size}')
        self.logg(
            1, f'time steps:          dt: {self.dt}, \t t: {self.t[0]} -> {self.t[-1]}, \t steps: {self.t.size}')
        self.logg(1, f'Initial state:       {self.T0}')
        self.logg(1, "T1 = T0 + ( A @ T0 + b )")
        self.logg(
            1, f'{self.T1.shape} = {self.T0.shape} + ( {(self.n,self.n)} @ {self.T0.shape} + {(self.n,)} )')
        self.logg(1, f'----------------------------------------------')

    def logg(self, lvl, txt, logger=print):
        """
        Logg when lvl >= self.logging
         - 0: nothing
         - 1: only initial setup
         - 2: time steps
         - 3: A and b
         - 4: everything
        """
        if self.logging >= lvl:
            logger(txt)

    def plot(self):
        """
        Plot the current state
        """
        # TODO
        ...

    def solve_next(self, method="cd"):
        """
        Calculate the next time step (T1)
        """
        if method == "cd":
            A, b = self.make_Ab()
            self.T1[...] = self.T0 + (self.dt/self.a) * (A @ self.T0 + b)

    def solve_all(self, method="cd"):
        """
        Iterate through from t0 -> tn
        """
        self.logg(1, "Iterating...",)
        for t in self.t:
            self.logg(2, f'- t = {t}')
            self.solve_next(method)
            self.save(self.T1)
            self.T0, self.T1 = self.T1, np.zeros(self.n)
        self.logg(1, "Finished",)
        self.logg(1, f'Final state: {self.T0}')

    def save(self, array):
        """
        save array to disk(self.filename)
        """
        # TODO
        ...

    def make_Ab(self):
        """
        Contruct A and b
        """

        # diagonal indicies
        [k0, k1, k2, k3, k4, k5, k6] = self.get_ks()
        ks = [k0, k1, k2, k3, k4, k5, k6]

        # ------- contruct all diagonals -------
        d = np.ones(self.n)

        C1 = self.b/self.dh**2 + self.c/(2*self.dh)
        C2 = self.b/self.dh**2 - self.c/(2*self.dh)
        C3 = 6*self.b/self.dh**2
        C4 = 2*self.dh/self.alpha

        d0 = -C3*d.copy()

        d1 = C1*d.copy()
        d2 = C1*d.copy()
        d3 = C1*d.copy()

        d4 = C2*d.copy()
        d5 = C2*d.copy()
        d6 = C2*d.copy()

        ds = [d0, d1, d2, d3, d4, d5, d6]

        # --------------- modify the boundaries ---------------
        # see project report
        # TODO:
        # [X] set C1+C2
        # [X] set 0
        # not sure if implemented correctly, need to validate with manufactored solutions
        # - tested with neuman boundary = 0 -> behaves correctly
        # - need to validate with non-zero values / functions

        d0[self.bis[:, 0]] -= (-self.bis[:, 1]*C2 +
                               self.bis[:, 1]*C1)*C4*self.beta

        i1 = self.diag_indicies(1)
        i2 = self.diag_indicies(2)
        i3 = self.diag_indicies(3)
        i4 = self.diag_indicies(4)
        i5 = self.diag_indicies(5)
        i6 = self.diag_indicies(6)

        d1[i1] = C1+C2
        d1[i1+k4] = 0
        d1 = d1[k1:]

        d2[i2] = C1+C2
        d2[i2+k5] = 0
        d2 = d2[k2:]

        d3[i3] = C1+C2
        d3[i3+k6] = 0
        d3 = d3[k3:]

        d4[i4+k4] = C1+C2
        d4[i1+k4] = 0
        d4 = d4[:k4]

        d5[i5+k5] = C1+C2
        d5[i2+k5] = 0
        d5 = d5[:k5]

        d6[i6+k6] = C1+C2
        d6[i3+k6] = 0
        d6 = d6[:k6]
        # -----------------------------------------------------
        A = diags(ds, ks)

        b = np.zeros(self.n)
        b[self.bis[:, 0]] = (-self.bis[:, 1]*C2 +
                             self.bis[:, 2]*C1)*C4*self.gamma

        self.logg(3, f'A = {A}')
        self.logg(3, f'b = {b}')
        return A, b

    # --------------- Helper methods for make_Ab ---------------

    def index_of(self, i, j, k):
        """
        Returns the 1D index from 3D cordinates
        """
        return i + self.I*j + self.I*self.J*k

    def get_ks(self):
        """
        Get the ks to use in diags(ds, ks) in self.make_Ab
        """
        k0 = 0
        k1 = self.index_of(1, 0, 0)
        k2 = self.index_of(0, 1, 0)
        k3 = self.index_of(0, 0, 1)
        k4 = self.index_of(-1, 0, 0)
        k5 = self.index_of(0, -1, 0)
        k6 = self.index_of(0, 0, -1)
        return k0, k1, k2, k3, k4, k5, k6

    def diag_indicies(self, bnd):
        """
        Finds the indicies of a secific boundary

        bnd:
        - 0: x = 0
        - 1: y = 0
        - 2: z = 0
        - 3: x = X
        - 4: y = Y
        - 5: z = Z

        Used to index the diagonals

        May only be run one time for each diagonal
        """

        i = (bnd == 4 and [self.I-1]) or (bnd == 1 and [0]) or range(self.I)
        j = (bnd == 5 and [self.J-1]) or (bnd == 2 and [0]) or range(self.J)
        k = (bnd == 6 and [self.K-1]) or (bnd == 3 and [0]) or range(self.K)
        # just ignore sort? it doesn't break without it
        return np.sort(np.array(self.index_of(*np.meshgrid(i, j, k))).T.reshape(-1))

    def find_border_indicies(self):
        """
        returns the indicies for every boundary node

        [[index,  # of start, # of end],...]

        only need to be run one time
        """
        indicies = np.zeros((self.border, 3), dtype=np.int16)
        tmp = 0
        for k in range(self.K):
            for j in range(self.J):
                for i in range(self.I):
                    if i == 0 or i == (self.I-1) or j == 0 or j == (self.J-1) or k == 0 or k == (self.K-1):
                        indicies[tmp] = np.array(
                            [self.index_of(i, j, k), *self.sum_start_and_end(i, j, k)])
                        tmp += 1
        return indicies

    def sum_start_and_end(self, i, j, k):
        """
        wacky way to count the different boundaries the node borders

        E.g: \n
        (0, 0, 0) -> [3, 0] \n
        (0, 0, Z) -> [2, 1] \n
        (0, 4, Z) -> [1, 1]
        """
        # pretend you didn't see this
        boundaries = ((i == 0 and "start") or (i == self.I-1 and "end"),
                      (j == 0 and "start") or (j == self.J-1 and "end"),
                      (k == 0 and "start") or (k == self.K-1 and "end"))

        start = 0
        end = 0
        for boundary in boundaries:
            start += 1 if boundary == "start" else 0
            end += 1 if boundary == "end" else 0
        return start, end

    # ----------------------------------------------------------
