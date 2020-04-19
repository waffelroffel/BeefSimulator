# beef_functions.py

import numpy as np
import BeefSimulator
from pathlib import Path
from typing import Union
import json


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


# Cooked from plotter class
def load_from_file( path: Path ):
    if not isinstance( path, Path ):
        path = Path( path )
    head_path = path.joinpath( "header.json" )
    temp_path = path.joinpath( "T.dat" )
    cons_path = path.joinpath( "C.dat" )
    
    header = None
    with open( head_path ) as f:
        header = json.load( f )
    
    dt = header[ "dt" ]
    dh = header[ "dh" ]
    
    dims = header[ "dims" ]
    shape = header[ "shape" ]
    t = np.linspace( header[ "t0" ], header[ "tn" ], shape[ 0 ] )
    x = np.linspace( dims[ "x0" ], dims[ "xn" ], shape[ 1 ] )
    y = np.linspace( dims[ "y0" ], dims[ "yn" ], shape[ 2 ] )
    z = np.linspace( dims[ "z0" ], dims[ "zn" ], shape[ 3 ] )
    
    T_data = np.memmap( temp_path,
                        dtype="float64",
                        mode="r",
                        shape=tuple( shape ) )
    C_data = np.memmap( cons_path,
                        dtype="float64",
                        mode="r",
                        shape=tuple( shape ) )
    
    t_jump = header[ "t_jump" ]
    
    return [ dt, dh, t, x, y, z, T_data, C_data, t_jump ]


# Collect data without initializing beef object
def collect_data( quantity: str, t: float, path: Path ):
    stuff = load_from_file( path )
    tj = stuff[ 8 ]
    if tj == -1:
        n = -1
    else:
        n = int( t / tj )
    if quantity == 'T':
        return stuff[ 6 ][ n ]
    elif quantity == 'C':
        return stuff[ 7 ][ n ]
