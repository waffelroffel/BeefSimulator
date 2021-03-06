import numpy as np
from BeefSimulator import BeefSimulator

'''
Make data for testing of plot etc.

With a pixel size of f.ex. 0.001 meters, if we assume the beef slice to have dimensions 10x15x20 cm
Then the dataset has dimensions 100*150*200

Import this file and find test data in a,b = make_testdata() where a, b are orthorhombic 3D numpy arrays

'''


def make_testdata( shape ) -> (np.array, np.array):
    data = np.ones( shape ) * 50  # Set all values to 50 (degrees?)
    
    # Sine variation in z-direction from 25 to 75, constant in xy-plane
    variation = np.zeros_like( data )
    for j in range( shape[ 2 ] ):
        variation[ :, :, j ] = 25 * np.sin( 2 * np.pi * j / shape[ 2 ] )
    data1 = data + variation
    
    # Hot 'bands' in the xy-plane, constant over z
    variation = np.zeros_like( data )
    for j in range( shape[ 0 ] ):
        for k in np.arange( -25, 25 ):
            variation[ j, 75 + int( 50 * np.sin( j / 6 ) ) + k, : ] = 25 - abs( k )
    data2 = data + variation
    
    return data1, data2


def make_manisol_1d( xd: int, td: int ) -> np.array:
    '''
    Boundary value problem:
    Solve T_t = a * T_x^2 (where f_q means partial derivative of f with respect to q) for x in [0,L], t in [0, inf)
    (1): T(0, t) = 0
    (2): T(L, t) = 0
    (3): T(x, 0) = sin(2*pi*x/L)		(Chosen for convenience)
    Only the n=2-Fourier component is non-zero due to B.C. (3) => T(x,t) = exp(-4*a*pi^2/L^2 * t) * sin(2*pi/L * x)
    Insert a = 1/8e6, L = 1

    :param xd: number of points discretising x
    :param td: number of points discretising t
    :return: The manifactured solution to the BVP described above
    '''
    
    # Analytic solution
    def T( x, t ): return np.exp( -4 * 1/8e6 * (np.pi)**2 * t ) * np.sin( 2 * np.pi * x )
    
    x = np.linspace( 0, 1, xd )
    delta_t = 0.1
    t = np.linspace( 0, td * delta_t, td )
    xx, tt = np.meshgrid( x, t )
    
    return T( xx, tt )


def make_manisol_3d( shape_d: list, filename: str ) -> None:
    '''
    Boundary value problem:
    Solve T_t = a * (T_x^2 + T_y^2 + T_z^2) for x in [0, Lx], y in [0, Ly], z in [0, Lz], t in [0, inf)
    (1.1): T(0,y,z,t) = 0		(1.2): T(Lx,y,z,t) = 0
    (2.1): T(x,0,z,t) = 0		(2.2): T(x,Ly,z,t) = 0
    (3.1): T(x,y,0,t) = T(x,y,Lz,t)			(3.2): T_z(x,y,0,t) = T_z(x,y,Lz,t)
    (4): T(x,y,z,0) = 3*sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*sin(4*pi*z/Lz)			(Chosen for convenience)
    Only the (nx=2, ny=2, nz=2)-Fourier component is non-zero due to B.C. (4) => ...
    T(x,y,z,t) = 3 * exp(-4*a*pi^2*(1/Lx^2+1/Ly^2+4/Lz^2) * t) * sin(2*pi/Lx * x) * sin(2*pi/Ly * y) * sin(4*pi/Lz * z)
    insert a = 1/8e6, Lx=Ly=Lz=1

    ### NB! (3.1) -> T(x,y,0,t) = 0 and (3.2) -> T(x,y,Lz,t) = 0 gives exactly the same solution!
    
    :param shape_d: tuple[int] of the numbers of points discretising t, x, y and z (Nt, Nx, Ny, Nz)
    :param filename: filename of data, something such as 'file.dat'
    :return: None. The manufactured solution to the BVP described above is saved to a file.
    '''
    
    # Analytic solution
    def T( x, y, z, t ): return 3 * np.exp( -4 * 1/8e6 * (np.pi)**2 * (1 + 1 + 4) * t ) * \
                                np.sin( 2 * np.pi * x ) * np.sin( 2 * np.pi * y ) * np.sin( 4 * np.pi * z )
    
    data = np.memmap( filename,
                      dtype='float64',
                      mode='w+',
                      shape=(shape_d) )
    
    td, xd, yd, zd = shape_d
    x = np.linspace( 0, 1, xd )
    y = np.linspace( 0, 1, yd )
    z = np.linspace( 0, 1, zd )
    dt = 0.1
    
    xx, yy, zz = np.meshgrid( x, y, z, indexing='ij' )
    
    for n in range( td ):
        data[ n ] = T( xx, yy, zz, n * dt )


#########################################################################################################
# HELPER FUNCTIONS #
D = 1
Lx, Ly, Lz = 1, 1, 1

xi = lambda x: np.sin( 2 * np.pi / Lx * x )
eta = lambda y: np.sin( 2 * np.pi / Ly * y )
zeta = lambda z: np.sin( 4 * np.pi / Lz * z )
tau = lambda t: np.exp( -4 * D * t * np.pi**2 * (1 / Lx**2 + 1 / Ly**2 + 4 / Lz**2) )

f_x0 = lambda y, z, t: 2 * np.pi / Lx * eta( y ) * zeta( z ) * tau( t )
f_xL = lambda y, z, t: 2 * np.pi / Lx * eta( y ) * zeta( z ) * tau( t )
f_y0 = lambda x, z, t: xi( x ) * 2 * np.pi / Ly * zeta( z ) * tau( t )
f_yL = lambda x, z, t: xi( x ) * 2 * np.pi / Ly * zeta( z ) * tau( t )
f_z0 = lambda x, y, t: xi( x ) * eta( y ) * 4 * np.pi / Lz * tau( t )
f_zL = lambda x, y, t: xi( x ) * eta( y ) * 4 * np.pi / Lz * tau( t )

'''
Diff eq has Neumann boundary conditions

dT/dx|_0  = f_x0
dT/dx|_Lx = f_xL
dT/dy|_0  = f_y0
dT/dy|_Ly = f_yL
dT/dz|_0  = f_z0
dT/dz|_Lz = f_zL
T(x,y,z,0) =  3*sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*sin(4*pi*z/Lz)

Gives the same manifactured solution as before. You may insert these functions as inputs in BeefSimulator
'''


#########################################################################################################


def make_Dirichlet_analytic_object( shape: list, td: int ) -> np.array:
    analyticSol = make_manisol_3d( shape, td )
    initial = analyticSol[ :, :, :, 0 ]
    np.save( f'3D_manisol_T_{shape[ 0 ]}_{shape[ 1 ]}_{shape[ 2 ]}_{td}.npy', analyticSol )
    return initial


def T_init( x, y, z, t ):
    return 3 * np.exp( -4 * 1e-3 * (np.pi)**2 * (1 + 1 + 4) * t ) * \
           np.sin( 2 * np.pi * x ) * np.sin( 2 * np.pi * y ) * np.sin( 4 * np.pi * z )


def make_Dirichlet_test_object( shape: list, td: int ) -> None:
    dh = 0.01
    dt = 0.1
    beefdims = [ [ 0, shape[ 0 ] * dh ], [ 0, shape[ 1 ] * dh ], [ 0, shape[ 2 ] * dh ], [ 0, td * dt ] ]
    # Notat til meg selv: bør a, b, c spesifiseres komponentvis eller nah?
    a = 1
    b = 1
    c = 0
    # Antar at dette henspeiler alle front og bak for alle x, y, z retninger - derfor 6 komponenter.
    alpha = [ 0, 0, 0, 0, 0, 0 ]
    beta = [ 1, 1, 1, 1, 1, 1 ]
    gamma = [ 0, 0, 0, 0, 0, 0 ]
    
    Beef = BeefSimulator( beefdims, a, b, c, alpha, beta, gamma, T_init, np.zeros( shape.append( td ) ), dh,
                          dt, filename=f'../data/3D_testsol_T_{shape[ 0 ]}_{shape[ 1 ]}_{shape[ 2 ]}_{td}.npy',
                          logging=1, bnd_types=[ 'd', 'd', 'd', 'd', 'd', 'd' ] )
    Beef.solve_all( )
    Beef.plot( Beef.tn )


'''
NB:
Jeg har definert shape til å være [20, 30, 40]
Beef-objektet rapporterer dette til å være [1, 21, 31, 41]
Noen som har kommentarer ang. dette?
'''

if __name__ == '__main__':
    # Create sample manufactured solutions
    h = 0.5
    dt = 0.1
    x_m = 75
    y_m = 150
    z_m = 175
    
    t = np.arange( 0, 7.0 + dt, dt )
    x = np.arange( 0, x_m + h, h )
    y = np.arange( 0, y_m + h, h )
    z = np.arange( 0, z_m + h, h )
    
    d_shape = (len( t ), len( x ), len( y ), len( z ))
    print( d_shape )
    
    manu_file = 'manu_sol_temp.dat'
    make_manisol_3d( d_shape, manu_file )
