logging = {
    "init": True,  # print the configuration of beefsimulator
    "init_state": False,  # print the initial state of T
    "stage": True,  # print the different stages of beefsimulator
    "tn": True,  # print the current time step
    "Ab": False,  # print the A matrix and b vector
    "final": False  # print the final state of T
}

# numpy logspace
dt_list = [ 1e-3, 4.16189726e-4, 1.73213888e-44, 7.20898408e-5, 3.00030511e-5 ]
dh_list = [ 0.01, 0.00562341, 0.00316228, 0.00177828, 0.001 ]
dt_default = 0.0001
dh_default = 0.003

# If you want length 10, use these instead
# dh_list = [ 0.01, 0.00774264, 0.00599484, 0.00464159, 0.00359381, 0.00278256, 0.00215443, 0.0016681 , 0.00129155, 0.001 ]
# dt_list = [ 1e-3, 6.77323522e-4, 4.58767154e-4, 3.10733784e-4, 2.10467301e-4, 1.42554454e-4, 9.65554847e-5, 6.53993010e-5, 4.42964849e-5, 3.00030511e-5 ]

# NB!!! - Must be changed for each dataset
dt = dt_list[0] # = dt_list[i]
dh = dh_default # = dh_list[i]

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
