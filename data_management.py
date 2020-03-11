# data_management.py
#
# Functions to read/write data to csv

import numpy as np
import pandas as pd


# Overwrite = False, Write = True
def write_csv(arr: np.array, filename: str, wO: bool):
    wO = wO * 'a' + (1 - wO) * 'w'

    arr = np.hstack(arr[:, ...])
    pd.DataFrame(arr).to_csv('data/'+filename+'.csv', mode=wO, sep=';',  header=False)


# dim = [x-len, y-len, z-len, t-len]
# t-len = (t_start, t_end)
# return Beef in timeslots determined by t-len
def read_csv(filename: str, dim: np.array):
    df = pd.read_csv('data/'+filename+'.csv', sep=';', dtype=np.float, skiprows=0)
    Q = df.to_numpy()[:, 1:, ...]
    B = np.vsplit(Q, len(dim[3]))
    # Liker ikke loopen
    for i in dim[3]:
        B[i] = np.hsplit(B[i], dim[1])
    return B


