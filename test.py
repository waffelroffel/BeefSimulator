import numpy as np
from BeefSimulator import BeefSimulator
from scipy.sparse import diags
from configs.T_conf import T_conf
from configs.C_conf import C_conf
from configs.conf import conf
from auxillary_functions import u_w

if __name__ == "__main__":
    bs = BeefSimulator(conf, T_conf, C_conf)

    #np.savetxt("initial.csv", bs.T0)
    bs.solve_all()
    #np.savetxt("final.csv", bs.T0)

    tt = np.linspace(conf["t0"], conf["tn"], 11)

    for t in tt:
        bs.plot(t, 'T', z=0.5)

    """
    dims = conf["dims"]
    x = np.linspace(dims["x0"], dims["xn"], int(dims["xlen"]/conf["dh"])+1)
    y = np.linspace(dims["y0"], dims["yn"], int(dims["ylen"]/conf["dh"])+1)
    z = np.linspace(dims["z0"], dims["zn"], int(dims["zlen"]/conf["dh"])+1)
    xx, yy, zz = np.meshgrid(x, y, z)

    T = T_conf["analytic"]
    analytic = T(xx, yy, zz, conf["tn"]).flatten()
    analytic[bs.d_bnd_indices] = 0

    print(bs.T0)
    print(analytic)

    np.savetxt("test.csv", bs.T0)
    np.savetxt("analytic.csv", analytic)
    """
