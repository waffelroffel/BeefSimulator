from BeefSimulator import BeefSimulator
from configs.verify_decoupled.T_conf import T_conf
from configs.verify_decoupled.C_conf import C_conf
from configs.verify_decoupled.conf import conf

bs = BeefSimulator(conf, T_conf, C_conf)
bs.solver()
