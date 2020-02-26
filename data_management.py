# data_management.py
#
# Functions to read/write data to csv

import numpy as np
import pandas as pd


def write_csv(arr: np.array, filename: str):
    pd.DataFrame(arr).to_csv('data/'+filename+'.csv', sep=';')


def read_csv(filename: str):
    df = pd.read_csv('data/'+filename+'.csv', sep=';', dtype=np.float)
    return df.to_numpy()


# test
a = read_csv('testdata')
print(a)
print(type(a[0, 0]))
