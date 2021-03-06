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
