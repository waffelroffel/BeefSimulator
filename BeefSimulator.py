import numpy as np
import scipy.sparse as sp
import Plotting.BeefPlotter as BP
from pathlib import Path
import auxillary_functions as af
import json


class BeefSimulator:
    def __init__(self, conf, T_conf, C_conf, cmp_with_analytic=False):
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
        - [X] construct A and b
        - [X] make it work with only direchet boundary: alpha = 0
        - [X] change a, b, c, alpha, beta, gamma to functions
        - [X] implement u_w
        - [ ] validate with manufactured solution
        - [X] implement C (concentration)
        - [ ] T and C coupled
        - [X] add plotter
        - [ ] add data management
        - [X] change logging to a config (dict)
        """
        self.pre_check(conf, T_conf, C_conf)
        self.setup_geometry(conf, T_conf, C_conf)
        self.setup_indices(conf, T_conf, C_conf)
        self.setup_TC(conf, T_conf, C_conf)
        self.setup_mesh(conf, T_conf, C_conf)
        self.setup_files(conf, T_conf, C_conf)
        self.setup_additionals(conf, T_conf, C_conf)
        self.initial_logg(conf, T_conf, C_conf)

    def setup_geometry(self, conf, T_conf, C_conf):
        self.dh = conf["dh"]
        self.dt = conf["dt"]

        dims = conf["dims"]
        self.x = np.linspace(dims["x0"], dims["xn"],
                             int(dims["xlen"] / self.dh) + 1)
        self.y = np.linspace(dims["y0"], dims["yn"],
                             int(dims["ylen"] / self.dh) + 1)
        self.z = np.linspace(dims["z0"], dims["zn"],
                             int(dims["zlen"] / self.dh) + 1)
        self.t = np.linspace(conf["t0"], conf["tn"],
                             int(conf["tlen"] / self.dt) + 1)

        self.t_jump = conf["t_jump"]
        t_steps = 1 if self.t_jump == - \
            1 else int(self.t.size / self.t_jump) + 1

        self.space = (self.x.size, self.y.size, self.z.size)
        self.shape = (t_steps, self.x.size, self.y.size, self.z.size)
        self.I, self.J, self.K = self.shape[1:]
        self.num_nodes = self.I * self.J * self.K
        self.num_nodes_inner = (self.I - 2) * (self.J - 2) * (self.K - 2)
        self.num_nodes_border = self.num_nodes - self.num_nodes_inner

    def setup_indices(self, conf, T_conf, C_conf):
        # indices for the direchet nodes in T
        self.T_direchets = self.get_direchet_indices(T_conf["bnd_types"])
        # indices for the direchet nodes in C
        self.C_direchets = self.get_direchet_indices(C_conf["bnd_types"])
        # indices for the boundary nodes
        self.boundaries = self.find_border_indicies()
        # diagonal indices for make_Ab
        self.ks = self.get_ks()

    def setup_TC(self, conf, T_conf, C_conf):
        # Defines the PDE and boundary conditions for T
        self.T_a = self.wrap(T_conf["pde"]["a"])
        self.T_b = self.wrap(T_conf["pde"]["b"])
        self.T_c = self.wrap(T_conf["pde"]["c"])
        self.T_alpha = self.wrap(T_conf["bnd"]["alpha"])
        self.T_beta = self.wrap(T_conf["bnd"]["beta"])
        self.T_gamma = self.wrap(T_conf["bnd"]["gamma"])
        self.T_initial = self.wrap(T_conf["initial"])

        # Defines the PDE and boundary conditions for C
        self.C_a = self.wrap(C_conf["pde"]["a"])
        self.C_b = self.wrap(C_conf["pde"]["b"])
        self.C_c = self.wrap(C_conf["pde"]["c"])
        self.C_alpha = self.wrap(C_conf["bnd"]["alpha"])
        self.C_beta = self.wrap(C_conf["bnd"]["beta"])
        self.C_gamma = self.wrap(C_conf["bnd"]["gamma"])
        self.C_initial = self.wrap(C_conf["initial"])

        self.uw = T_conf["uw"] or C_conf["uw"]

    def setup_mesh(self, conf, T_conf, C_conf):
        xx, yy, zz = np.meshgrid(self.x, self.y, self.z)
        self.ii = [xx, yy, zz, self.t[0]]

    def setup_files(self, conf, T_conf, C_conf):
        self.T1 = np.zeros(self.num_nodes)
        self.T0 = np.zeros(self.num_nodes)
        self.T0[...] = self.T_initial(self.ii)

        self.C1 = np.zeros(self.num_nodes)
        self.C0 = np.zeros(self.num_nodes)
        self.C0[...] = self.C_initial(self.ii)

        self.path = Path("data").joinpath(conf["folder"])
        if not self.path.exists():
            self.path.mkdir()

        self.H_file = self.path.joinpath("header.json")
        self.save_header(conf)

        self.T_file = self.path.joinpath("T.dat")
        self.T_data = np.memmap(
            self.T_file, dtype="float64", mode="w+", shape=self.shape)

        self.C_file = self.path.joinpath("C.dat")
        self.C_data = np.memmap(
            self.C_file, dtype="float64", mode="w+", shape=self.shape)

    def setup_additionals(self, conf, T_conf, C_conf):
        self.plotter = BP.Plotter(self, name=Path(
            "data").joinpath(conf["folder"]), save_fig=True)
        self.logging = conf["logging"]

    def initial_logg(self, conf, T_conf, C_conf):
        self.logg("stage", f'SETUP FINISHED. LOGGING...')
        self.logg("init", f'----------------------------------------------')
        self.logg("init", f'Logging level:       {self.logging}')
        self.logg("init", f'Shape:               {self.shape}')
        self.logg("init", f'Total nodes:         {self.num_nodes}')
        self.logg("init", f'Inner nodes:         {self.num_nodes_inner}')
        self.logg("init", f'Boundary nodes:      {self.num_nodes_border}')
        self.logg("init",
                  f'x linspace:          dx: {self.dh}, \t x: {self.x[ 0 ]} -> {self.x[ -1 ]}, \t steps: {self.x.size}')
        self.logg("init",
                  f'y linspace:          dy: {self.dh}, \t y: {self.y[ 0 ]} -> {self.y[ -1 ]}, \t steps: {self.y.size}')
        self.logg("init",
                  f'z linspace:          dz: {self.dh}, \t z: {self.z[ 0 ]} -> {self.z[ -1 ]}, \t steps: {self.z.size}')
        self.logg("init",
                  f'time steps:          dt: {self.dt}, \t t: {self.t[ 0 ]} -> {self.t[ -1 ]}, \t steps: {self.t.size}')
        self.logg("init", "T1 = T0 + ( A @ T0 + b )")
        self.logg("init",
                  f'{self.T1.shape} = {self.T0.shape} + ( {(self.num_nodes, self.num_nodes)} @ {self.T0.shape} + {(self.num_nodes,)} )')
        self.logg("init", f'----------------------------------------------')
        self.logg("init_state", f'Initial state:       {self.T0}')

    def save_header(self, conf: dict):
        header = conf.copy()
        header.pop("logging")
        header["shape"] = self.shape
        with open(self.H_file, "w+") as f:
            json.dump(header, f)

    # -------------------- Solver --------------------

    def set_vars(self, id):
        cond = id == "T"
        self.a = self.T_a if cond else self.C_a
        self.b = self.T_b if cond else self.C_b
        self.c = self.T_c if cond else self.C_c
        self.alpha = self.T_alpha if cond else self.C_alpha
        self.beta = self.T_beta if cond else self.C_beta
        self.gamma = self.T_gamma if cond else self.C_gamma
        self.direchets = self.T_direchets if cond else self.C_direchets

    def solver(self, method="cd"):
        """
        Iterate through from t0 -> tn
        solve for both temp. and conc.
        """

        del self.T_data
        del self.C_data
        self.T_data = np.memmap(
            self.T_file, dtype="float64", mode="w+", shape=self.shape)
        self.C_data = np.memmap(
            self.C_file, dtype="float64", mode="w+", shape=self.shape)

        self.logg("stage", "Iterating...", )
        for step, t in enumerate(self.t):
            # save each "step"
            if self.t_jump != -1 and step % self.t_jump == 0:
                i = int(step / self.t_jump)
                self.T_data[i] = self.T0.reshape(self.shape[1:])
                self.C_data[i] = self.C0.reshape(self.shape[1:])
                self.T_data.flush()
                self.C_data.flush()

            self.u = self.uw(self.T0, self.C0, *self.space, self.dh)

            self.ii[3] = t
            self.logg("tn", f'- t = {t}')

            self.set_vars("T")
            self.solve_next(self.T0, self.T1, method)
            self.T0, self.T1 = self.T1, np.empty(self.num_nodes)

            self.set_vars("C")
            self.solve_next(self.C0, self.C1, method)
            self.C0, self.C1 = self.C1, np.empty(self.num_nodes)

        # save last step (anyway)
        self.T_data[-1] = self.T0.reshape(self.shape[1:])
        self.C_data[-1] = self.C0.reshape(self.shape[1:])
        self.T_data.flush()
        self.C_data.flush()

        self.logg("stage", "Finished", )
        self.logg("final", f'Final state: {self.T0}')

    def solve_next(self, U0, U1, method="cd"):
        """
        Calculate the next time step (T1)
        """
        if method == "cd":
            A, b = self.make_Ab()
            U1[...] = U0 + (self.dt / self.a(self.ii)) * (A @ U0 + b)

            U1[self.direchets] = self.gamma(self.ii)[self.direchets] / \
                self.beta(self.ii)[self.direchets]

    def make_Ab(self):
        """
        Construct A and b
        """
        # ------- contruct all diagonals -------

        bh2 = self.b(self.ii) / self.dh**2
        c2h = self.c(self.ii) / (2 * self.dh)

        u = self.u
        ux = u[:, 0]
        uy = u[:, 1]
        uz = u[:, 2]

        C1_x = bh2 + c2h * ux
        C2_x = bh2 - c2h * ux
        C1_y = bh2 + c2h * uy
        C2_y = bh2 - c2h * uy
        C1_z = bh2 + c2h * uz
        C2_z = bh2 - c2h * uz

        C_u = np.array([C1_x, -C2_x, C1_y, -C2_y, C1_z, -C2_z]).transpose()

        C3 = 6 * self.b(self.ii) / self.dh**2

        _alpha = self.alpha(self.ii).copy()
        _alpha[self.direchets] = 1  # dummy
        _alpha[_alpha == 0] = 1  # dummy
        C4 = 2 * self.dh / _alpha

        d = np.ones(self.num_nodes)
        d0, d1, d2, d3, d4, d5, d6 = [-C3 * d,
                                      C1_x * d, C1_y * d,
                                      C1_z * d, C2_x * d,
                                      C2_y * d, C2_z * d]

        # --------------- modify the boundaries ---------------
        # see project report
        # TODO:
        # [X] set C1+C2
        # [X] set 0
        # not sure if implemented correctly, need to validate with manufactored solutions
        # - tested with neuman boundary = 0 -> behaves correctly
        # - need to validate with non-zero values / functions

        # add u_w to prod
        prod = af.dotND(
            self.boundaries[:, 1:], C_u[self.boundaries[:, 0]], axis=1)
        d0[self.boundaries[:, 0]] -= prod * C4[self.boundaries[:, 0]] * \
            self.beta(self.ii)[self.boundaries[:, 0]]

        ks = self.ks
        k0, k1, k2, k3, k4, k5, k6 = self.ks
        i1, i2, i3, i4, i5, i6 = [self.diag_indicies(i + 1) for i in range(6)]

        d1[i1] = (C1_x + C2_x)[i1]
        d1[i1 + k4] = 0
        d1 = d1[k1:]

        d2[i2] = (C1_y + C2_y)[i2]
        d2[i2 + k5] = 0
        d2 = d2[k2:]

        d3[i3] = (C1_z + C2_z)[i3]
        d3[i3 + k6] = 0
        d3 = d3[k3:]

        d4[i4 + k4] = (C1_x + C2_x)[i4 + k4]
        d4[i1 + k4] = 0
        d4 = d4[:k4]

        d5[i5 + k5] = (C1_y + C2_y)[i5 + k5]
        d5[i2 + k5] = 0
        d5 = d5[:k5]

        d6[i6 + k6] = (C1_z + C2_z)[i6 + k6]
        d6[i3 + k6] = 0
        d6 = d6[:k6]

        # -----------------------------------------------------
        ds = [d0, d1, d2, d3, d4, d5, d6]
        A = sp.diags(ds, ks)

        b = np.zeros(self.num_nodes)

        prod = af.dotND(
            self.boundaries[:, 1:], C_u[self.boundaries[:, 0]], axis=1)
        b[self.boundaries[:, 0]] = prod * C4[self.boundaries[:, 0]] * \
            self.gamma(self.ii)[self.boundaries[:, 0]]

        self.logg("Ab", f'A = {A}')
        self.logg("Ab", f'b = {b}')
        return A, b

    # -------------------- Logger --------------------

    def logg(self, lvl, txt, logger=print):
        """
        See config/conf.py for details
        """
        if self.logging[lvl]:
            logger(txt)

    # -------------------- Plotter --------------------

    def plot(self, t, id, x=None, y=None, z=None):
        """
        Plot the current state
        param id: Either 'T' or 'C'
        params x, y, z: perpendicular cross-section of beef to plot.
        """
        if id == 'T':
            self.plotter.show_heat_map(self.T_data, t, id, x, y, z)
        elif id == 'C':
            self.plotter.show_heat_map(self.C_data, t, id, x, y, z)
        else:
            raise ValueError(
                'Trying to aquire a quantity that does not exist.')

    # -------------------- Retrieve data from time call --------------------

    def get_data_from_time(self, id: str, t: float) -> np.array:
        '''
        Generalised retrieval of datapoint, currently supports T and C
        :param id: Either 'T' or 'C'
        :param t: time
        :return: the value of T or C at time t
        '''
        n = int(t / self.dt)
        if n < self.t.size:
            if id == 'T':
                return self.T_data[n]
            elif id == 'C':
                return self.C_data[n]
            else:
                raise ValueError(
                    'Trying to aquire a quantity that does not exist.')
        else:
            raise IndexError(f'Trying to access time step no. {n} of the beef object, but it only has {self.t.size} '
                             f'entries!')

    def get_T(self, t: float) -> np.array:
        '''
        :param t: time
        :return: T at time t
        '''
        return self.get_data_from_time('T', t)

    def get_C(self, t: float) -> np.array:
        '''
        :param t: time
        :return: C at time t
        '''
        return self.get_data_from_time('C', t)

    # --------------- Helper methods for make_Ab & make_Cd ---------------

    def index_of(self, i, j, k):
        """
        Returns the 1D index from 3D coordinates
        """
        return i + self.I * j + self.I * self.J * k

    # Unnecessary? TODO: remove?
    def index_of_inverse(self, p):
        """
        Returns the 3D index from 1D coordinate
        """
        k = p // (self.I * self.J)
        j = (p - k * self.I * self.J) // self.I
        i = (p - k * self.I * self.J - j * self.I)
        return i, j, k

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
        Finds the indices of a specific boundary

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

        i = (bnd == 4 and [self.I - 1]) or (bnd == 1 and [0]) or range(self.I)
        j = (bnd == 5 and [self.J - 1]) or (bnd == 2 and [0]) or range(self.J)
        k = (bnd == 6 and [self.K - 1]) or (bnd == 3 and [0]) or range(self.K)
        # just ignore sort? it doesn't break without it
        return np.sort(np.array(self.index_of(*np.meshgrid(i, j, k))).T.reshape(-1))

    def get_direchet_indices(self, bnd_types):
        uniques = set()

        bnd_lst = []
        for qqq, bnd in enumerate(bnd_types):
            if bnd == "d":
                bnd_lst.append(self.diag_indicies(qqq + 1))

        for qq in bnd_lst:
            for q in qq:
                uniques.add(q)

        # TODO: can remove sorted
        return sorted(list(uniques))

    def find_border_indicies(self, new=False):
        """
        returns the indices for every boundary node

        [[index, x0, xn, y0, yn, z0, zn],...]

        E.g: (0,Y,2) \n
        [[index, 1, 0, 0, 1, 0, 0],...]
        """
        indicies = np.zeros((self.num_nodes_border, 7), dtype=np.int16)
        tmp = 0
        for k in range(self.K):
            for j in range(self.J):
                for i in range(self.I):
                    if i == 0 or i == (self.I - 1) or j == 0 or j == (self.J - 1) or k == 0 or k == (self.K - 1):
                        indicies[tmp] = np.array(
                            [self.index_of(i, j, k), *self.sum_start_and_end(i, j, k, True)])
                        tmp += 1
        return indicies

    def sum_start_and_end(self, i, j, k, new=False):
        """
        returns which borders the (i,j,k) node lies on

        either 0 or 1 on:
        [x0, xn, y0, yn, z0, zn]

        E.g: \n
        (0, 0, 0) -> [1, 0, 1, 0, 1, 0] \n
        (0, 0, Z) -> [1, 0, 1, 0, 0, 1] \n
        (0, 4, Z) -> [1, 0, 0, 0, 0, 1]
        """
        x0 = int(i == 0)
        xn = int(i == self.I - 1)
        y0 = int(j == 0)
        yn = int(j == self.J - 1)
        z0 = int(k == 0)
        zn = int(k == self.K - 1)
        return x0, xn, y0, yn, z0, zn

    # --------------- Misc. -------------------------------------------

    def wrap(self, fun):
        def _wrap(ii):
            res = fun(*ii) if callable(fun) else fun
            return res.flatten() if isinstance(res, np.ndarray) else np.ones(ii[1].size) * res
        return _wrap

    def pre_check(self, conf, T_conf, C_conf):
        def _check_T_or_C(conf, prefix):
            assert conf["pde"]["a"] is not None, f'{prefix}: a should not be None'
            assert conf["pde"]["b"] is not None, f'{prefix}: b should not be None'
            assert conf["pde"]["c"] is not None, f'{prefix}: c should not be None'
            assert conf["bnd"]["alpha"] is not None, f'{prefix}: alpha should not be None'
            assert conf["bnd"]["beta"] is not None, f'{prefix}: beta should not be None'
            assert conf["bnd"]["gamma"] is not None, f'{prefix}: gamma should not be None'
            assert conf["uw"] is not None, f'{prefix}: uw should not be None'
            assert 0 <= len(conf["bnd_types"]) <= 6, \
                f'{prefix}: bnd_types should be of length 0-6, got length={len( conf[ "bnd_types" ] )}'
            assert conf["initial"] is not None, f'{prefix}: initial should not be None'

        assert conf["dh"] > 0, f'conf: dh should be >0, got dh={conf[ "dh" ]}'
        assert conf["dt"] > 0, f'conf: dt should be >0, got dt={conf[ "dt" ]}'

        t_jump = conf["t_jump"]
        assert type(t_jump) == int,\
            f'conf: t_jump should be integer, got type(t_jump)={type(t_jump)}'
        assert t_jump > 0 or t_jump == -1,\
            f'conf: dt should be >0 or -1, got dt={t_jump}'

        assert conf["tlen"] > 0, f'conf: tlen should be >0, got tlen={conf[ "tlen" ]}'
        assert conf["tn"] > conf["t0"],\
            f'conf: tn should be >t0, got t0={conf[ "t0" ]} and tn={conf[ "tn" ]}'

        dims = conf["dims"]
        assert dims["xlen"] > 0, f'conf: xlen should be >0, got xlen={dims[ "xlen" ]}'
        assert dims["xn"] > dims["x0"],\
            f'conf: xn should be >x0, got x0={dims[ "x0" ]} and xn={dims[ "xn" ]}'
        assert dims["ylen"] > 0, f'conf: ylen should be >0, got ylen={dims[ "ylen" ]}'
        assert dims["yn"] > dims["y0"],\
            f'conf: yn should be >y0, got y0={dims[ "y0" ]} and yn={dims[ "yn" ]}'
        assert dims["zlen"] > 0, f'conf: zlen should be >0, got zlen={dims[ "zlen" ]}'
        assert dims["zn"] > dims["z0"],\
            f'conf: zn should be >z0, got z0={dims[ "z0" ]} and zn={dims[ "zn" ]}'

        _check_T_or_C(T_conf, "T")
        _check_T_or_C(C_conf, "C")

    def __str__(self):
        return "Sorry can't do... yet"
