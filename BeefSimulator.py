import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
#import Plotting.BeefPlotter as BP


class BeefSimulator:
    def __init__(self, dims, a, b, c, alpha, beta, gamma, dh=0.01, dt=0.1, initial=0, filename="data.csv"):
        """
        dims: [ [x_start, x_len], [y_start, y_len], ... , [t_start, t_len] ]

        pde: a*dT_dt = b*T_nabla + c*T_gradient \n
        boundary: alpha*T_gradient + beta*T = gamma

        dh: stepsize in each dim
        dt: time step

        initial: scalar or function with parameter (x,y,z,t)
        """
        #self.plotter = BP.Plotter(self)
        self.filename = filename

        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.dh = dh
        self.dt = dt

        steps = [(s, s+l, l//dh+1) for s, l in dims]

        self.x = np.linspace(*steps[0])
        self.y = np.linspace(*steps[1])
        self.z = np.linspace(*steps[2])
        self.t = np.linspace(*steps[3])

        self.shape = (self.x.size, self.y.size, self.z.size)
        self.I, self.J, self.K = self.shape
        self.n = self.I * self.J * self.K

        # send it through self.index_of before use
        xx, yy, zz = np.meshgrid(self.x, self.y, self.z)

        self.T1 = np.zeros(self.n)
        self.T0 = np.zeros(self.n)
        self.T0[...] = initial(xx, yy, zz) if callable(initial) else initial
        self.save([])  # header
        self.save(self.T0)

        print(f'Shape of Prism: {self.shape}')
        print(f'Time steps: {len(self.t)}')
        print(f'x linspace: {self.x}')
        print(f'y linspace: {self.y}')
        print(f'z linspace: {self.z}')
        # print(f'Initial condition: {self.T0}')

    def solve_next(self, method="cd"):
        if method == "cd":
            A, b = self.make_Ab()
            print(f'T0 + (A @ T0 + b)')
            print(
                f'{self.T0.shape} + ({A.shape} @ {self.T0.shape} + {b.shape})')
            # transpose A ?
            self.T1 = self.T0 + (self.dt/self.a) * (A @ self.T0 + b)

    def solve_all(self, method="cd"):
        print("Iterating")
        for t in self.t:
            print(f'{t=}')
            self.solve_next(method)
            self.save(self.T1)
            self.T0, self.T1 = self.T1, np.zeros(self.shape)
        print("Finished")

    def save(self, array):
        ...

    def make_Ab(self):
        I, J, K = self.shape
        n = I*J*K
        print(f'Total nodes:    {I*J*K}')
        print(f'Inner nodes:    {(I-2)*(J-2)*(K-2)}')
        print(f'Boundary nodes: {I*J*K-(I-2)*(J-2)*(K-2)}')
        bis = self.find_border_indicies()

        # shorter variables
        b = self.b
        c = self.c
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        dh = self.dh

        # diagonal indicies
        [k0, k1, k2, k3, k4, k5, k6] = self.get_ks()
        ks = [k0, k1, k2, k3, k4, k5, k6]

        # ------- contruct all diagonals -------
        d = np.ones(n)

        C0 = -6*b/dh**2
        C1 = b/dh**2 + c/(2*dh)
        C2 = b/dh**2 - c/(2*dh)

        d0 = C0*d.copy()

        d1 = C1*d.copy()
        d2 = C1*d.copy()
        d3 = C1*d.copy()

        d4 = C2*d.copy()
        d5 = C2*d.copy()
        d6 = C2*d.copy()

        ds = [d0, d1, d2, d3, d4, d5, d6]

        # modify the boundaries
        # see project report
        # TODO:
        # [X] set C1-C2
        # [ ] set 0

        d0[bis[:, 0]] -= (bis[:, 1]*C2 + bis[:, 1]*C1)*2*dh*beta/alpha

        i1 = self.diag_indicies(1)
        d1[i1] = C1-C2
        # d1[i1-1] = 0
        d1 = d1[k1:]

        i2 = self.diag_indicies(2)
        d2[i2] = C1-C2

        i3 = self.diag_indicies(3)
        d3[i3] = C1-C2

        i4 = self.diag_indicies(4)
        d4[i4] = C1-C2
        d4[i1-1] = 0

        i5 = self.diag_indicies(5)
        d5[i5] = C1-C2

        i6 = self.diag_indicies(6)
        d6[i6] = C1-C2

        # --------------------------------------
        # print(d1)
        # print(d4)
        A = diags(ds, ks).todense()
        #print(len(np.diag(A, k1)))

        b = np.zeros(n)
        b[bis[:, 0]] = (bis[:, 1]*C2 + bis[:, 2]*C1)*2*dh*gamma/alpha

        # print(f'{A=}')
        # print(f'{b=}')
        return A, b

    def index_of(self, i, j, k):
        return i + self.I*j + self.I*self.J*k

    def get_ks(self):
        k0 = 0
        k1 = self.index_of(1, 0, 0)
        k2 = self.index_of(0, 1, 0)
        k3 = self.index_of(0, 0, 1)
        k4 = self.index_of(-1, 0, 0)
        k5 = self.index_of(0, -1, 0)
        k6 = self.index_of(0, 0, -1)
        return k0, k1, k2, k3, k4, k5, k6

    def diag_indicies(self, boundary):
        indicies = []
        kk = [
            self.K-1] if boundary == 6 else [0] if boundary == 3 else range(self.K)
        jj = [
            self.J-1] if boundary == 5 else [0] if boundary == 2 else range(self.J)
        ii = [
            self.I-1] if boundary == 4 else [0] if boundary == 1 else range(self.I)
        for k in kk:
            for j in jj:
                for i in ii:
                    indicies.append(self.index_of(i, j, k))
        return np.array(indicies)

    def find_border_indicies(self):
        """
        returns the indicies for every boundary node

        [[index, # of start, # of end],...]

        only need to be run one time
        """
        indicies = np.zeros((self.I*self.J*self.K-(self.I-2)
                             * (self.J-2)*(self.K-2), 3), dtype=np.int16)
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
        (0,0,0) -> [3,0] \n
        (0,0,Z) -> [2,1] \n
        (0,4,Z) -> [1,1]
        """
        boundaries = ((i == 0 and "start") or (i == (self.I-1) and "end"),
                      (j == 0 and "start") or (
            j == (self.J-1) and "end"),
            (k == 0 and "start") or (k == (self.K-1) and "end"))

        start = 0
        end = 0
        for boundary in boundaries:
            start += 1 if boundary == "start" else 0
            end += 1 if boundary == "end" else 0
        return start, end
