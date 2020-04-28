import convergence_test
from Beef_experiment import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble=r'\usepackage{siunitx}')


folder_name = 'data/conv_oven_t780_dt0.0001_dh0.0013'
calc = []


for t in timelist1:
    data = convergence_test.get_data_from_foldername(folder_name, t)
    # [1] = Temperature
    # [-1], [-1] = max x and y directions
    # int(z_len/2) = half z direction
    zlen = len(data[0][0][0][0]);
    T = data[1][-1][-1][int((zlen-1)/2)]
    calc.append(T)


plt.figure()
# plt.title("Beef A and Beef B")
plt.scatter(timelist1, T1, label=r"Beef A")
plt.scatter(timelist2, T2, label=r"Beef B")
plt.scatter(timelist1, calc, label=r"Calculated values")
plt.xlabel(r"$t$[s]")
plt.ylabel(r"Temperature $T$ in center [C]")
plt.legend()
plt.grid()
plt.gcf()
plt.savefig(fname='data/compareExperiment.pdf', format='pdf')
plt.show()
