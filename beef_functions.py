# beef_functions.py

import numpy as np
import BeefSimulator


def compare_beefs( beef1: BeefSimulator, beef2: BeefSimulator, t: float, pprty: str ) -> float:
    '''
    Calculates the (Frobernius) norm of the difference in T or C of beef2-beef1 at time t
    :param beef1: The first beef object
    :param beef2: The second beef object
    :param t: Time at which the data is retrieved
    :param pprty: The property we are measuring. Must be either 'T' or 'C'
    :return: The norm of T or C in beef2-beef1
    '''
    
    if (pprty == 'T'):
        return np.linalg.norm( beef2.get_T( t ) - beef1.get_T( t ) )
    
    elif (pprty == 'C'):
        return np.linalg.norm( beef2.get_C( t ) - beef1.get_C( t ) )
    
    else:
        raise ValueError( "No proper property to compare given. pprty has to either be 'T' or 'C'." )
