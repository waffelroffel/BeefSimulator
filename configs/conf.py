# conf
# conv_oven

logging = {
    "init": True,  # print the configuration of beefsimulator
    "init_state": False,  # print the initial state of T
    "stage": True,  # print the different stages of beefsimulator
    "tn": True,  # print the current time step
    "Ab": False,  # print the A matrix and b vector
    "final": False  # print the final state of T
}

folder = "conv_oven_test"

dh = 1e-3
dt = 1e-5

# -1: only the last
<<<<<<< HEAD
t_jump = 100 # Save progress each 100 time step

t0 = 0
tlen = 1e-1
=======
t_jump = -1  # Save progress each full second

t0 = 0
tlen = 0.001
>>>>>>> b746a06dadf5d77db4606bad596d94a11d7c26b5
tn = t0+tlen

# Actual beef size split symmetrically along xy-axes. Implement symmetric B.C.s
# to compensate.

x0 = 0
xlen = 0.075
xn = x0+xlen

y0 = 0
ylen = 0.04
yn = y0+ylen

z0 = 0
zlen = 0.055
zn = z0+zlen

conf = {
    "dh": dh,
    "dt": dt,
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
    "folder": folder,
    "t_jump": t_jump
}
