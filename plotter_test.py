from Plotting.BeefPlotter import Plotter
from pathlib import Path

if __name__ == "__main__":
    bp = Plotter(name=Path("data/convtest_T_dt0.001_dh0.003"))
    bp.save_fig = False
    bp.show_heat_map2('T', 0, X=[0.015, 0.030, 0.045, 0.060, 0.075])
#     for i in range(14):
#         bp.show_heat_map2("T", i * 0.2, X=[0, 0.25, 0.5, 0.75, 1])
