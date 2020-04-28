logging = {
    "init": True,  # print the configuration of beefsimulator
    "init_state": False,  # print the initial state of T
    "stage": True,  # print the different stages of beefsimulator
    "tn": False,  # print the current time step
    "Ab": False,  # print the A matrix and b vector
    "final": False  # print the final state of T
}

# save to data/"folder name"
folder = "example"

# time and space step
dh = 0.1
dt = 0.001

# specify the saving interval for the timestep
# -1: only the last
# 0: save every step
# 10: save every tenth step
t_jump = 1

# define start and lenght of t dimension
t0 = 0
tlen = 1
tn = t0+tlen

# define start and lenght of x dimension
x0 = 0
xlen = 1
xn = x0+xlen

# define start and lenght of y dimension
y0 = 0
ylen = 1
yn = y0+ylen

# define start and lenght of z dimension
z0 = 0
zlen = 1
zn = z0+zlen

# no need to touch this
# it collect all the values to a easy to pass dict
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
