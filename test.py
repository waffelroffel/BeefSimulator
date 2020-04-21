import numpy as np
from BeefSimulator import BeefSimulator
from scipy.sparse import diags
from configs.T_conf import T_conf
from configs.C_conf import C_conf
from configs.conf import conf
from auxillary_functions import u_w

if __name__ == "__main__":
    bs = BeefSimulator(conf, T_conf, C_conf)

    bs.solver()

    n = int(conf["tlen"] / conf["dt"]) + 1
    print(n)

    tt = np.linspace(conf["t0"], conf["tn"], n)

    # Latex support for matplotlib does not work if installation requirements are not met.
    # See https://matplotlib.org/3.2.1/tutorials/text/usetex.html for more information.
    # It also takes slightly longer to render a Latex figure than otherwise.
    bs.plotter.set_latex(True)

    bs.plot('T', 0, z=[0, 0.01, 0.02, 0.03, 0.04])
