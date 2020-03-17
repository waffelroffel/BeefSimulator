import numpy as np
from BeefSimulator import BeefSimulator
from scipy.sparse import diags


dims = [[0, 40], [0, 40], [0, 40], [0, 10]]
bs = BeefSimulator(dims=dims, a=1, b=1, c=0, alpha=1, beta=0,
                   gamma=0, initial=1, dh=1, dt=0.01, logging=1)
bs.solve_all()
