from BeefSimulator import BeefSimulator
from configs.config_library.conv_oven.T_conf import T_conf
from configs.config_library.conv_oven.C_conf import C_conf
from configs.config_library.conv_oven.conf import conf
import datetime

if __name__ == "__main__":
    start = datetime.datetime.now()
    print("Running BeefSimulator for convection oven.")
    print("Started at:", start, "\n")
    bs = BeefSimulator(conf, T_conf, C_conf)

    bs.solver()
    end = datetime.datetime.now()
    print("Convection oven - Started at:", start, "\tFinished at:", end)
    print("Total elapsed time:", end - start)
    print()
