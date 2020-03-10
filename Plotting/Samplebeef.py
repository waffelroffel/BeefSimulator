import numpy as np

class Beef:
    def __init__(self, U, C, t, x, y, z):
        self.U = U
        self.C = C
        
        self.t = t
        self.x = x
        self.y = y
        self.z = z

        self.dt = t[1] - t[0]
        self.h = x[1] - x[0]