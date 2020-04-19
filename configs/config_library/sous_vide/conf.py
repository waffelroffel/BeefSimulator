## conf
## sous_vide

logging = {
    "init": True,  # print the configuration of beefsimulator
    "init_state": False,  # print the initial state of T
    "stage": True,  # print the different stages of beefsimulator
    "tn": True,  # print the current time step
    "Ab": False,  # print the A matrix and b vector
    "final": False  # print the final state of T
}

dh = 1e-3
dt = 1e-4

# -1: only the last step is saved
t_jump = 10000 # Save progress each 10 seconds

t0 = 0
tlen = 780 #13 min
tn = t0+tlen

folder = f"sous_vide_{tlen}sec"

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
