logging = {
    "init": False,  # print the configuration of beefsimulator
    "init_state": False,  # print the initial state of T
    "stage": False,  # print the different stages of beefsimulator
    "tn": False,  # print the current time step
    "Ab": False,  # print the A matrix and b vector
    "final": False  # print the final state of T
}

folder = "github"

dh = 0.1
dt = 0.01

# -1: only the last
t_jump = 1

t0 = 0
tlen = 0.1
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
