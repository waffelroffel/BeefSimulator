from Plotting.BeefPlotter import Plotter

if __name__ == "__main__":
    bp = Plotter(name="data/verify_decoupled")

    tt = [0, 1]
    bp.show_heat_map2("T", T=tt, Y=0.5)
    bp.show_heat_map2("C", T=tt, Y=0.3)
