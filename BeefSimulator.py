import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import Plotting.BeefPlotter as BP
from data_management import write_csv
from pathlib import Path
import os
import auxillary_functions as af
import constants as const


class BeefSimulator:
    def __init__(self, dims, a, b, c, alpha, beta, gamma, initial,initial_C, dh=0.01, dt=0.1, filename="data/data", logging=1, bnd_types=[]):
        """
        dims: [ [x_start, x_len], [y_start, y_len], ... , [t_start, t_len] ]

        pde: a*dT_dt = b*T_nabla + c*T_gradient

        boundary: alpha*T_gradient + beta*T = gamma

        dh: stepsize in each dim

        dt: time step

        initial: scalar or function with parameter (x,y,z,t)

        bnd_types:
         - d: direchet (only this do something)
         - n: neumann
         - r: robin

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
        - [X] change a, b, c, alpha, beta, gamma to functions
        - [ ] validate with manufactored solution
        - [X] implement C (concentration)
        - [ ] T and C coupled
        - [X] add plotter
        - [ ] add data management
        - [ ] add another logg level between 1 and 2 for linspaces and initial state
        """
        def _wrap(fun):
            def wrap(ii):
                res = fun(*ii) if callable(fun) else fun
                return res.flatten() if isinstance(res, np.ndarray) else res
            return wrap

        # Defines the PDEs and boundary conditions
        self.a = _wrap(a)
        self.b = _wrap(b)
        self.c = _wrap(c)
        self.alpha = _wrap(alpha)
        self.beta = _wrap(beta)
        self.gamma = _wrap(gamma)
        self.initial = _wrap(initial)

        self.dh = dh
        self.dt = dt

        xyz_steps = [(s, s+l, int(l/dh+1)) for s, l in dims[:-1]]
        t_steps = (dims[-1][0], dims[-1][0]+dims[-1][1], int(dims[-1][1]/dt+1))

        self.x = np.linspace(*xyz_steps[0])
        self.y = np.linspace(*xyz_steps[1])
        self.z = np.linspace(*xyz_steps[2])
        self.t = np.linspace(*t_steps)

        self.shape = (t_steps[2], self.x.size, self.y.size, self.z.size)
        self.I, self.J, self.K = self.shape[1:]
        self.n = self.I * self.J * self.K
        self.inner = (self.I-2) * (self.J-2) * (self.K-2)
        self.border = self.n-self.inner

        # ---------- refactor ----------
        uniques = set()

        bnd_lst = []
        for qqq, bnd in enumerate(bnd_types):
            if bnd == "d":
                bnd_lst.append(self.diag_indicies(qqq+1))

        for qq in bnd_lst:
            for q in qq:
                uniques.add(q)

        self.direchet_bnds = sorted(list(uniques))
        # ---------- refactor ----------

        # rename: the 1D indicies for all the boundary points
        self.bis = self.find_border_indicies()

        xx, yy, zz = np.meshgrid(self.x, self.y, self.z)
        self.tn = self.t[0]
        self.ii = (xx, yy, zz, self.tn)

        self.T1 = np.zeros(self.n)
        self.T0 = np.zeros(self.n)
        self.T0[...] = self.initial(self.ii)

        self.C1 = np.zeros(self.n)
        self.C0 = np.zeros(self.n)
        self.C0[...] = initial_C(xx, yy, zz, self.tn) if callable(
            initial_C) else initial_C  # currently doesn't support function

        self.filename = filename
        self.H_file = Path(self.filename + '_header.csv')
        self.H_file.open('w+').close()

        self.T_file = Path(self.filename + '_temp.dat')
        self.T_data = np.memmap(self.T_file, 
                                dtype='float64', 
                                mode= 'r+' if self.T_file.exists() else 'w+', 
                                shape=self.shape)

        self.C_file = Path(self.filename + '_cons.dat')
        self.C_data = np.memmap(self.C_file, 
                                dtype='float64', 
                                mode='r+' if self.C_file.exists() else 'w+', 
                                shape=self.shape)
       
        #self.save([], self.H_file, 'csv')  # save header data: dims, time steps, ...
        #self.save(self.T0, self.T_file, 'npy')

        self.plotter = BP.Plotter(self, name=filename, save_fig=True)

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

    def plot(self, t = None, x = None, y = None, z = None):
        """
        Plot the current state
        x, y, or z: perpendicular cross-section of beef to plot.
        """
        t_ = self.tn if t == None else t
        self.plotter.show_heat_map(self.T_data, t_, x, y, z)

    def solve_next(self, method="cd"):
        """
        Calculate the next time step (T1)
        """
        if method == "cd":
            A, b = self.make_Ab()
            self.T1[...] = self.T0 + \
                (self.dt/self.a(self.ii)) * (A @ self.T0 + b)

            self.T1[self.direchet_bnds] = (self.gamma(
                self.ii)/self.beta(self.ii))[self.direchet_bnds]
    
    def solve_all(self, method="cd"):
        """
        Iterate through from t0 -> tn
        """
        
        self.logg(1, "Iterating...",)
        for i in range(len(self.t)):
            self.tn = i * self.dt
            self.logg(2, f'- t = {self.tn}')
            self.solve_next(method)
            self.T_data[i] = self.T1.reshape(self.shape[1:])
            self.T0, self.T1 = self.T1, np.zeros(self.n)
        self.logg(1, "Finished",)
        self.logg(1, f'Final state: {self.T0}')
    
    def save(self, array, file, ext):
        """
        save array to disk(self.filename)

        array: array to save. E.g. temperature or concentration array for a given timestep.

        file: file/path to save to. E.g. self.H_file, self.T_file, self.C_file

        ext: file extension. E.g 'npy' or 'csv'
        """
        if (ext == 'npy'):
            with file.open('ab') as f:
                np.save(f, array.reshape((1, self.shape[1], self.shape[2], self.shape[3])))
        elif (ext == 'csv'):
            write_csv(array, file, False)
        else:
            raise ValueError("ext must either be 'npy' or 'csv'.")

    def make_Ab(self,):
        """
        Contruct A and b
        """

        # diagonal indicies
        [k0, k1, k2, k3, k4, k5, k6] = self.get_ks()
        ks = [k0, k1, k2, k3, k4, k5, k6]

        # ------- contruct all diagonals -------
        d = np.ones(self.n)

        C1 = self.b(self.ii)/self.dh**2 + self.c(self.ii)/(2*self.dh)
        C2 = self.b(self.ii)/self.dh**2 - self.c(self.ii)/(2*self.dh)
        C3 = 6*self.b(self.ii)/self.dh**2

        _alpha = self.alpha(self.ii).copy()
        _alpha[self.direchet_bnds] = 1  # dummy
        _alpha[_alpha == 0] = 1  # dummy
        C4 = 2*self.dh/_alpha

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

        d0[self.bis[:, 0]] -= (-self.bis[:, 1]*C2[self.bis[:, 0]] +
                               self.bis[:, 1]*C1[self.bis[:, 0]])*C4[self.bis[:, 0]]*self.beta(self.ii)[self.bis[:, 0]]

        d0[self.direchet_bnds] = 1

        i1 = self.diag_indicies(1)
        i2 = self.diag_indicies(2)
        i3 = self.diag_indicies(3)
        i4 = self.diag_indicies(4)
        i5 = self.diag_indicies(5)
        i6 = self.diag_indicies(6)

        d1[i1] = (C1+C2)[i1]
        d1[i1+k4] = 0
        d1 = d1[k1:]

        d2[i2] = (C1+C2)[i2]
        d2[i2+k5] = 0
        d2 = d2[k2:]

        d3[i3] = (C1+C2)[i3]
        d3[i3+k6] = 0
        d3 = d3[k3:]

        d4[i4+k4] = (C1+C2)[i4+k4]
        d4[i1+k4] = 0
        d4 = d4[:k4]

        d5[i5+k5] = (C1+C2)[i5+k5]
        d5[i2+k5] = 0
        d5 = d5[:k5]

        d6[i6+k6] = (C1+C2)[i6+k6]
        d6[i3+k6] = 0
        d6 = d6[:k6]

        # -----------------------------------------------------
        A = diags(ds, ks)

        b = np.zeros(self.n)
        b[self.bis[:, 0]] = (-self.bis[:, 1]*C2[self.bis[:, 0]] +
                             self.bis[:, 2]*C1[self.bis[:, 0]])*C4[self.bis[:, 0]]*self.gamma(self.ii)[self.bis[:, 0]]

        self.logg(3, f'A = {A}')
        self.logg(3, f'b = {b}')
        return A, b

    # ----------------------- Concentration solver ------------------------------

    def solve_next_C(self, method="cd"):
        """
        Calculate the next time step (C1)
        """
        if method == "cd":
            C, d = self.make_Cd()
            self.C1[...] = self.C0 + (self.dt/(2 * self.dh**2) * (C @ self.C0 + d))

    def solve_all_C(self, method="cd"):
        """
        Iterate through from t0 -> tn
        """
        # Clear data before solving for all C.
        self.C_data = np.memmap(self.C_file, dtype='float64', mode='w+', shape=self.shape)

        self.logg(1, "Iterating...",)
        for i in range(len(self.t)):
            self.tn = i * self.dt
            self.logg(2, f'- t = {self.tn}')
            self.solve_next_C(method)
            self.C_data[i] = self.C1.reshape(self.shape[1:])
            self.C0, self.C1 = self.C1, np.zeros(self.n)
        self.logg(1, "Finished",)
        self.logg(1, f'Final state: {self.C0}')

    def make_Cd(self):
        """
        Construct C and d for Concentration equation
        """
        # diagonal indices
        [k0, k1, k2, k3, k4, k5, k6] = self.get_ks()
        ks = [k0, k1, k2, k3, k4, k5, k6]

        # ------- construct all diagonals -------
        d = np.ones(self.n)

        # TODO:
        # Fix \nabla u_w, currently placeholder
        # !DOES NOT WORK!
        D1 = 2 * self.dh * af.u_w(self.T0, self.C0) + const.D
        D2 = - 2 * self.dh * af.u_w(self.T0, self.C0) + const.D
        D3 = 2 * self.dh * np.gradient(af.u_w(self.T0, self.C0)) - 6 * const.D

        d0 = D3 * d.copy()

        d1 = D1 * d.copy()
        d2 = D1 * d.copy()
        d3 = D1 * d.copy()

        d4 = D2 * d.copy()
        d5 = D2 * d.copy()
        d6 = D2 * d.copy()

        ds = [d0, d1, d2, d3, d4, d5, d6]

        # TODO:
        # --------------- modify the boundaries ---------------

        # TODO:
        # -----------------------------------------------------
        C = diags(ds, ks)

        d = np.zeros(self.n)
        # ehm:
        #d[self.bis[:, 0]] = (-self.bis[:, 1] * C2 +
        #                     self.bis[:, 2] * C1) * C4 * self.gamma

        self.logg(3, f'C = {C}')
        self.logg(3, f'd = {d}')
        return C, d

    # --------------- Helper methods for make_Ab & make_Cd ---------------

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
        - 1: x = 0
        - 2: y = 0
        - 3: z = 0
        - 4: x = X
        - 5: y = Y
        - 6: z = Z

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
