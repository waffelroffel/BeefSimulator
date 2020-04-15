from Plotting.BeefPlotter import Plotter

if __name__ == "__main__":
    bp = Plotter(name="data/beef1")
    bp.show_heat_map2(0.0, "T", x=0.5)
