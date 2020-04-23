from BeefSimulator import BeefSimulator
from configs.config_library.frying_pan.T_conf import T_conf
from configs.config_library.frying_pan.C_conf import C_conf
from configs.config_library.frying_pan.conf import conf
import datetime

if __name__ == "__main__":
    start = datetime.datetime.now()
    print("Running BeefSimulator for frying pan.")
    print("Started at:", start, "\n")
    bs = BeefSimulator(conf, T_conf, C_conf)

    bs.solver()
    end = datetime.datetime.now()
    print("Frying pan - Started at:", start, "\tFinished at:", end)
    print("Total elapsed time:", end - start)
    print()
