from Plotting.BeefPlotter import Plotter

if __name__ == "__main__":
    bp = Plotter(name="data/conv_oven_test")
    bp.show_heat_map2("T", 0, X=[0.01, 0.011, 0.012, 0.013])
