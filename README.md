# BeefSimulator

Eksperter i Team - Matematikk innen anvendelser - Team 2

---

## How to use BeefSimulator

### 1. Setup the configs

A template on the different config files can be found in config/

Only need to touch these three files:

- [conf.py](configs/conf.py)
- [T_conf.py](configs/T_conf.py)
- [C_conf.py](configs/C_conf.py)

**Geometry of the beef**: [conf.py](configs/conf.py)

**PDE and boundary conditions for Temperature**: [T_conf.py](configs/T_conf.py)

**PDE and boundary conditions for Concentration**: [T_conf.py](configs/C_conf.py)

Governed by the two equations:

- pde: ![pde](https://render.githubusercontent.com/render/math?math=%5Ca%20%5Cfrac%7BdT%7D%7Bdt%7D%20%3D%20b%20%5Cnabla%5E2%20T%20%2B%20c%20%5Ctextbf%7Bu%7D%20%5Cnabla%20T)
- boundary: ![bnd](https://render.githubusercontent.com/render/math?math=%5Calpha%20%5Cnabla%20T%20%2B%20%5Cbeta%20%5Ctextbf%7Bu%7D%20T%20%3D%20%5Cgamma)
- (currently missing u_w in pde and boundary, will update later)

Notes:

- a, b, c: must be scalars
- alpha, beta and gamma can be scalar, vector or function that returns scalar or array
- **u**: is a vector for the water velocity
- the positive direction is defined to be inward into the cube -> gamma must then be set accordingly

A example with descriptions of the configuration files can be found under _configs/example_.

---

### 2. Run BeefSimulator

To run and create data:

```python
from BeefSimulator import BeefSimulator
from configs.T_conf import T_conf
from configs.C_conf import C_conf
from configs.conf import conf

bs = BeefSimulator(conf, T_conf, C_conf) # initialize a BeefSimulator with the given configs, will save the header data in data/"folder_name"/header.json
bs.solver() # iterate all time steps for Temperature and Concentration and save the data in data/"folder_name"/
```

---

### 3. Plotting the results

The BeefSimulator class has an BeefPlotter member object which one can use to plot the data inherent in the BS class. Currently, one can simply use the BS member function `plot()` with the appropriate arguments to make use of most of the BeefPlotter features.
To view a perpendicular crossection of the beef at a given time step:

```python
bs.plot(0, x=0.5, 'T') # Temperature heatmap yz-slice at x=0.5 and t=0
bs.plot(0.01, y=0.5, 'T') # Temperature heatmap xz-slice at y=0.5 and t=0.01
bs.plot(0.1, z=0.5, 'C') # Concentration xy-slice at z=0.5 and t=0.1
```

In order to view multiple crossections from an isometric projection (WIP), simply set the arguments to be a list:

```python
bs.plot(0, x=[0.0, 0.5, 1.0], 'T') # Temperature heatmap yz-crossections at x=0.0, 0.5, and 1.0 at time t=0.
bs.plot(0.01, y=[0.5], 'T') # Temperature heatmap xz-crossections at y=0.5 at time t=0.01
bs.plotter.save_fig = False # Figures will not be saved to disk anymore (the plotter saves by default \)
bs.plot([0.0, 2.0, 4.0, 8.0], z=0.5, 'C') # Concentration heat xy-crossections at z=0.5 at times t=0.0, 2.0, 4.0, and 8.0.
```

The plots are saved in the same directory as the data is stored as pdf files. You can also enable LaTeX typesetting by including the line `bs.plotter.set_latex(True)` before plotting. The installation requirements must be fulfilled in order to make it work. See https://matplotlib.org/3.2.1/tutorials/text/usetex.html for more information.

---

#### Standalone plotting

To plot from precomputed results:

```python
bp = Plotter(name="path to data folder")
bp.show_heat_map2(0.0, "T", X=0.5) # cross section at x=0.5 at t=0
bp.show_heat_map2(0.0, "T", X=[0.5]) # cross section in 3D
bp.show_heat_map2(0.0, "T", X=[0.5, 0.4]) # multiple cross sections
```

Folder must contain:

- _header.json_
- _T.dat_
- _C.dat_

**tl;dr**: run [test_example.py](test_example.py) (will show a plot for each tenth of the time range specified in [example/conf.py](configs/example/conf.py))
