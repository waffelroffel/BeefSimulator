from BeefSimulator import BeefSimulator
from configs.example.T_conf import T_conf
from configs.example.C_conf import C_conf
from configs.example.conf import conf
from Plotting.BeefPlotter import Plotter

if __name__ == "__main__":
    # running solver
    bs = BeefSimulator(conf, T_conf, C_conf)
    bs.solver()
    # plot using BeefSimulator
    bs.plot("T", t=[0, 0.5, 1], x=[0.1, 0.4])

    # standalone plotting
    plt = Plotter(name="data/example")
    plt.show_heat_map2("T", T=[0, 0.5, 1], X=[0.1, 0.4])
