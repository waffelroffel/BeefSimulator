import numpy as np
from BeefSimulator import BeefSimulator

dims = [[0, 2], [0, 2], [0, 2], [0, 2]]
bs = BeefSimulator(dims=dims, a=1, b=1, c=0, alpha=1, beta=0,
                   gamma=0, initial=1, dh=0.05, dt=0.000001, logging=1)
bs.solve_all()
