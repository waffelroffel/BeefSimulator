logging = {
    "init": True,  # print the configuration of beefsimulator
    "init_state": False,  # print the initial state of T
    "stage": True,  # print the different stages of beefsimulator
    "tn": True,  # print the current time step
    "Ab": False,  # print the A matrix and b vector
    "final": False  # print the final state of T
}

# Example
dt_list = [ 0.001, 0.000316228, 0.0001, 0.000031623, 0.00001 ]
dh_list = [ 0.03, 0.0094868331, 0.003, 0.00094868331, 0.0003 ]
dt_default = 0.0001
dh_default = 0.003

# Must be changed for each dataset
dt = dt_default
dh = dh_default
folder = f"convtest_T_dt{dt:.2g}_dh{dh:.2g}"

# -1: only the last
t_jump = -1

t0 = 0
tlen = 10
tn = t0+tlen

x0 = 0
xlen = 0.075
xn = x0+xlen

y0 = 0
ylen = 0.039
yn = y0+ylen

z0 = 0
zlen = 0.054
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
