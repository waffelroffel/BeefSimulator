# convergence_test.py

from BeefSimulator import BeefSimulator
import numpy as np
import beef_functions
import matplotlib.pyplot as plt
import typing

# Advanced type hints, just for fun
beef_list = typing.List[ BeefSimulator ]
float_list = typing.List[ float ]


# A function that parses config file and initializes beef objects with the data corresponding to the parsed folder names
def initialize_conv_beefs( conf ) -> beef_list:
    du_list: float_list = conf[ 'du_list' ]
    N: int = len( du_list )
    beef_names = [ f'beef{n}' for n in range( N ) ]
    calc_beefs = [
        ... ]  # Lists of beefs to be initialized with appropriate config files (may become unnecessary if
    # functionality to extract data without configing the beef object is ever added
    


def produce_conv_plotdata( calc_beefs: beef_list, analytic_beefs: beef_list, t: float, du_list: float_list,
                           u_id: str, pprty: str ) -> np.array:
    '''
    Produces np.array that can be fed to plt.plot(data[0], data[1]) to plot the results of the convergence test
    :param calc_beefs: list of numerically iterated beef objects. Must all have produced data up to time t
    :param analytic_beefs: list of analytic beef objects at time t. If u_id='dt', the beef list has length 1. If,
    u_id='dh', this list has the same length as du_list, and each analytic solution has the same dimension as the
    corresponding calc_beef-objects
    :param t: the time at which the beef objects are evaluated
    :param du_list: the steps of du used (du may refer to dt or dh=dx,dy,dz)
    :param u_id: identifying which parameter we are convergence testing. Must be either 'dt' or 'dh'
    :param pprty: which property we are comparing. Must be either 'T' or 'C'
    :return: The calculated absolute norms for the differential values specified in du_list after iterating to time t
    '''
    
    N = len( du_list )
    norms = np.zeros( N )
    
    if (pprty == 'T' or pprty == 'C'):
        
        if u_id == 'dt':
            # Analytic data list has length 1
            for n in range( N ):
                norms[ n ] = abs( beef_functions.compare_beefs( calc_beefs[ n ], analytic_beefs[ 0 ], t, pprty ) )
        
        elif u_id == 'dh':
            # Analytic data has the same length as du_list
            for n in range( N ):
                norms[ n ] = abs( beef_functions.compare_beefs( calc_beefs[ n ], analytic_beefs[ n ], t, pprty ) )
        
        else:
            raise ValueError( f'I can only produce convergence tests for "dt" and "dh", but you fed me "{u_id}"!' )
    
    else:
        raise ValueError( f'I can only test convergence for "T" and "C", but you fed me "{pprty}"!' )
    
    # This data may immediately be plotted data[0] vs data[1]
    return np.array( [ du_list, norms ] )


# Much pseudo code
# TODO: Implement stuff that does not exist.
def plot_convergencetest( data: np.array, config_file: str ):
    plt.plot( data[ 0 ], data[ 1 ], marker='o', lw=5 )
    plt.xlabel( config_file.get_xlabel( ) )
    plt.ylabel( config_file.get_ylabel( ) )
    plt.show( )
    plt.savefig( fname=config_file.get_figfilename( ), format='pdf' )
