import convergence_test
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# mpl.rc('text', usetex=True)
# mpl.rc('text.latex', preamble=r'\usepackage{siunitx}')


folder_name_coup = 'data/conv_oven_t780_dt0.0001_dh0.0013'
folder_name_un = 'data/uncoupled_t780_dt0.0001_dh0.0013'
mid_un = []
mid_coup = []
edge_un = []
edge_coup = []

timelist=np.linspace(0, 13*60, 79)


for t in timelist:
    data_un = convergence_test.get_data_from_foldername(folder_name_un, t)
    data_coup = convergence_test.get_data_from_foldername(folder_name_coup, t)
    # [1] = Temperature
    # [-1], [-1] = max x and y directions
    # int(z_len/2) = half z direction
    zlen = len(data_un[0][0][0][0]);
    M_un = data_un[1][-1][-1][int((zlen-1)/2)]
    M_coup = data_coup[1][-1][-1][int((zlen-1)/2)]
    E_un = data_un[1][0][0][int((zlen-1)/2)]
    E_coup = data_coup[1][0][0][int((zlen-1)/2)]
    mid_un.append(M_un)
    mid_coup.append(M_coup)
    edge_un.append(E_un)
    edge_coup.append(E_coup)


plt.figure()
# plt.title("Beef A and Beef B")
plt.semilogy(timelist, np.array(mid_coup)-np.array(mid_un), label=r"Centre", lw=2.5)
plt.semilogy(timelist, np.array(edge_coup)-np.array(edge_un), label=r"Edge", lw=2.5)
plt.xlabel(r"$t$[s]")
plt.ylabel(r"$T_{coup}-T_{un}$ [C]")
plt.legend()
plt.grid()
plt.gcf()
plt.savefig(fname='data/compareUncoupled.pdf', format='pdf')
plt.show()
