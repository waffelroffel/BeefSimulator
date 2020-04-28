from Plotting.BeefPlotter import Plotter
from pathlib import Path

if __name__ == "__main__":
    bp = Plotter(name=Path("data/pos-conv_oven_t780_dt0.0001_dh0.0013"))
    bp.save_fig = True
    bp.set_latex(True)
    for i in range(14):
        bp.show_heat_map2("T", i * 60, X=[0.00, 0.015, 0.030, 0.045])
        bp.show_heat_map2("C", i * 60, X=[0.00, 0.015, 0.030, 0.045])
