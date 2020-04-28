from BeefSimulator import BeefSimulator
from configs.github.T_conf import T_conf
from configs.github.C_conf import C_conf
from configs.github.conf import conf

if __name__ == "__main__":
    bs = BeefSimulator(conf, T_conf, C_conf)
    bs.solver()
