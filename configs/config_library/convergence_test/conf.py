logging = {
    "init": True,  # print the configuration of beefsimulator
    "init_state": False,  # print the initial state of T
    "stage": True,  # print the different stages of beefsimulator
    "tn": True,  # print the current time step
    "Ab": False,  # print the A matrix and b vector
    "final": False  # print the final state of T
}

folder = "conv_dt_T_1_1e-3"

dh = 1e-2
dt = 1e-3

# -1: only the last
t_jump = -1

t0 = 0
tlen = 0.3
tn = t0+tlen

x0 = 0
xlen = 1
xn = x0+xlen

y0 = 0
ylen = 1
yn = y0+ylen

z0 = 0
zlen = 1
zn = z0+zlen

conf = {
    "dh": dh,
    "dt": dt,
    "t_jump": t_jump,
    "t0": t0,
    "tlen": tlen,
    "tn": tn,
    "dims": {
        "x0": x0,
        "xlen": xlen,
        "xn": xn,
        "y0": y0,
        "ylen": ylen,
        "yn": yn,
        "z0": z0,
        "zlen": zlen,
        "zn": zn,
    },
    "logging": logging,
    "folder": folder
}
