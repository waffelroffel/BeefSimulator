from Plotting.BeefPlotter import Plotter
from pathlib import Path

if __name__ == "__main__":
    bp = Plotter(name=Path("data/frying_pan_t780_dt0.0001_dh0.003"))
    bp.save_fig = False
    # bp.show_heat_map2("T", 0, X=[0.015, 0.030, 0.045, 0.060, 0.075])
    for i in range(13):
        bp.show_heat_map2("T", i * 60, X=[0.015, 0.030, 0.045, 0.060, 0.075])
