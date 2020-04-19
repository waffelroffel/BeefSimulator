from Plotting.BeefPlotter import Plotter

if __name__ == "__main__":
    bp = Plotter(name="data/conv_oven_test_3")
    bp.show_heat_map2("C", 10, Z=[0, 0.011, 0.022, 0.033, 0.044, 0.055])

