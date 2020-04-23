from Plotting.BeefPlotter import Plotter
from pathlib import Path

if __name__ == "__main__":
    bp = Plotter(name=Path("data/verify_decoupled"))
    bp.save_fig = True
    for i in range(14):
        bp.show_heat_map2("T", i * 0.2, X=[0, 0.25, 0.5, 0.75, 1])
