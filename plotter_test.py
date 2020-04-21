from Plotting.BeefPlotter import Plotter
from pathlib import Path

if __name__ == "__main__":
    bp = Plotter(name=Path("data/new_conv_oven_780sec"))
    bp.save_fig = True
    for i in range(14):
        bp.show_heat_map2("C", i * 60, X=[0, 0.015, 0.030, 0.045, 0.055])
