# BeefSimulator

Eksperter i Team - Matematikk innen anvendelser - Team 2

---

---

## How to use BeefSimulator

### 1. Setup the configs

_Currently only Temperature is set to work_

A template on the different config files can be found in config/

Only need to touch these three files:

- [conf.py](configs/conf.py)
- [T_conf.py](configs/T_conf.py)
- [C_conf.py](configs/C_conf.py)

**Geometry of the beef**: [conf.py](configs/conf.py)

- recommended to keep logging unchanged
- keep dt at least 100 times less than dh
- no need to touch the last conf dictionary

**PDE and boundary conditions for Temperature**: [T_conf.py](configs/T_conf.py)

Governed by the two equations:

- pde: ![pde](https://render.githubusercontent.com/render/math?math=a%20%5Cfrac%7BdT%7D%7Bdt%7D%20%3D%20b%20%5Cnabla%5E2%20T%20%2B%20c%20%5Cnabla%20T)
- boundary: ![bnd](https://render.githubusercontent.com/render/math?math=%5Calpha%20%5Cnabla%20T%20%2B%20%5Cbeta%20T%20%3D%20%5Cgamma)
- (currently missing u_w in pde and boundary, will update later)

Notes:

- If **alpha** is set to 0 or a list/array of 0, then **bnd_types** must be ["d", "d", "d", "d", "d", "d"]
- If **alpha** is non-zero, then **bnd_types** must be []
- Keep uw (water velocity ) as 1 for now
- a, b, c, alpha, beta and gamma can be scalar, vector or function that returns scalar or array
- no need to touch the last conf dictionary

---

### 2. Run BeefSimulator

To run and create data:

```python
from BeefSimulator import BeefSimulator
from configs.T_conf import T_conf
from configs.C_conf import C_conf
from configs.conf import conf

bs = BeefSimulator(conf, T_conf, C_conf) # initialize a BeefSimulator with the given configs, will save the header data in data/"folder_name"/header.json
bs.solve_all() # iterate all time steps for Temperature and save the temperature data in data/"folder_name"/T.dat
```

To view a slice (plane) of the beef at a given time step:

```python
bs.plot(0, x=0.5) # yz slice at x=0.5 and t=0
bs.plot(0.01, y=0.5) # xz slice at y=0.5 and t=0.01
bs.plot(0.1, z=0.5) # xy slice at z=0.5 and t=0.1
```

**tl;dr**: run [test.py](test.py) (will show a plot for each tenth of the time range specified in [conf.py](configs/conf.py))

---

---

## How to use BeefPlotter

### 1. ...
