import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import Plotting.BeefPlotter as BP
from data_management import write_csv
from pathlib import Path
import auxillary_functions as af
import constants as const
import json


class BeefSimulator:
    def __init__( self, conf, T_conf, C_conf, cmp_with_analytic=False ):
        """
        dims: [ [x_start, x_len], [y_start, y_len], ... , [t_start, t_len] ]

        pde: a*dT_dt = b*T_nabla + c*T_gradient

        boundary: alpha*T_gradient + beta*T = gamma

        dh: stepsize in each dim

        dt: time step

        initial: scalar or function with parameter (x,y,z,t)

        bnd_types:
         - d: direchet (only this do something)
         - n: neumann
         - r: robin

        logging:
         - 0: nothing
         - 1: only initial setup and end state
         - 2: time steps
         - 3: A and b
         - 4: everything
        """
        """
        TODO:
        - [X] change 3D cordinates to 1D indexing: T1 = T0 + ( A @ T0 + b )
        - [X] construct A and b
        - [X] make it work with only direchet boundary: alpha = 0
        - [X] change a, b, c, alpha, beta, gamma to functions
        - [X] implement u_w
        - [ ] validate with manufactured solution
        - [X] implement C (concentration)
        - [ ] T and C coupled
        - [X] add plotter
        - [ ] add data management
        - [X] change logging to a config (dict)
        """
        self.pre_check( conf, T_conf, C_conf )
        
        def _wrap( fun ):
            # TODO: move the checks one level higher
            def wrap( ii ):
                res = fun( *ii ) if callable( fun ) else fun
                return res.flatten( ) if isinstance( res, np.ndarray ) else np.ones( ii[ 1 ].size ) * res
            
            return wrap
        
        # Defines the PDE and boundary conditions for T
        self.a = _wrap( T_conf[ "pde" ][ "a" ] )
        self.b = _wrap( T_conf[ "pde" ][ "b" ] )
        self.c = _wrap( T_conf[ "pde" ][ "c" ] )
        self.alpha = _wrap( T_conf[ "bnd" ][ "alpha" ] )
        self.beta = _wrap( T_conf[ "bnd" ][ "beta" ] )
        self.gamma = _wrap( T_conf[ "bnd" ][ "gamma" ] )
        self.initial = _wrap( T_conf[ "initial" ] )
        self.uw = T_conf[ "uw" ]
        
        # Defines the PDE and boundary conditions for C
        self.initial_C = _wrap( C_conf[ "initial" ] )
        
        self.dh = conf[ "dh" ]
        self.dt = conf[ "dt" ]
        
        dims = conf[ "dims" ]
        self.x = np.linspace( dims[ "x0" ], dims[ "xn" ],
                              int( dims[ "xlen" ] / self.dh ) + 1 )
        self.y = np.linspace( dims[ "y0" ], dims[ "yn" ],
                              int( dims[ "ylen" ] / self.dh ) + 1 )
        self.z = np.linspace( dims[ "z0" ], dims[ "zn" ],
                              int( dims[ "zlen" ] / self.dh ) + 1 )
        self.t = np.linspace( conf[ "t0" ], conf[ "tn" ],
                              int( conf[ "tlen" ] / self.dt ) + 1 )
        
        self.shape = (self.t.size, self.x.size, self.y.size, self.z.size)
        self.I, self.J, self.K = self.shape[ 1: ]
        self.n = self.I * self.J * self.K
        self.inner = (self.I - 2) * (self.J - 2) * (self.K - 2)
        self.border = self.n - self.inner
        
        self.d_bnd_indices = self.get_d_bnd_indices( T_conf[ "bnd_types" ] )
        
        # rename: the 1D indicies for all the boundary points
        self.bis = self.find_border_indicies( )
        
        xx, yy, zz = np.meshgrid( self.x, self.y, self.z )
        self.ii = [ xx, yy, zz, self.t[ 0 ] ]
        
        self.T1 = np.zeros( self.n )
        self.T0 = np.zeros( self.n )
        self.T0[ ... ] = self.initial( self.ii )
        
        self.C1 = np.zeros( self.n )
        self.C0 = np.zeros( self.n )
        self.C0[ ... ] = self.initial_C( self.ii )
        
        self.base = Path( "data" )
        self.path = self.base.joinpath( conf[ "folder" ] )
        self.path.mkdir( ) if not self.path.exists( ) else ...
        
        self.H_file = self.path.joinpath( "header.json" )
        self.save_header( conf )
        
        self.T_file = self.path.joinpath( "T.dat" )
        self.T_data = self.memmap( self.T_file )
        
        self.C_file = self.path.joinpath( "C.dat" )
        self.C_data = self.memmap( self.C_file )
        
        self.plotter = BP.Plotter(
            self, name=self.path, save_fig=True )
        
        self.logging = conf[ "logging" ]
        
        self.logg( "stage", f'SETUP FINISHED. LOGGING...' )
        self.logg( "init", f'----------------------------------------------' )
        self.logg( "init", f'Logging level:       {self.logging}' )
        self.logg( "init", f'Shape:               {self.shape}' )
        self.logg( "init", f'Total nodes:         {self.n}' )
        self.logg( "init", f'Inner nodes:         {self.inner}' )
        self.logg( "init", f'Boundary nodes:      {self.border}' )
        self.logg( "init",
                   f'x linspace:          dx: {self.dh}, \t x: {self.x[ 0 ]} -> {self.x[ -1 ]}, \t steps: {self.x.size}' )
        self.logg( "init",
                   f'y linspace:          dy: {self.dh}, \t y: {self.y[ 0 ]} -> {self.y[ -1 ]}, \t steps: {self.y.size}' )
        self.logg( "init",
                   f'z linspace:          dz: {self.dh}, \t z: {self.z[ 0 ]} -> {self.z[ -1 ]}, \t steps: {self.z.size}' )
        self.logg( "init",
                   f'time steps:          dt: {self.dt}, \t t: {self.t[ 0 ]} -> {self.t[ -1 ]}, \t steps: {self.t.size}' )
        self.logg( "init", "T1 = T0 + ( A @ T0 + b )" )
        self.logg( "init",
                   f'{self.T1.shape} = {self.T0.shape} + ( {(self.n, self.n)} @ {self.T0.shape} + {(self.n,)} )' )
        self.logg( "init", f'----------------------------------------------' )
        self.logg( "init_state", f'Initial state:       {self.T0}' )
    
    def save_header( self, conf: dict ):
        header = conf.copy( )
        header.pop( "logging" )
        header[ "shape" ] = self.shape
        with open( self.H_file, "w" ) as f:
            json.dump( header, f )
    
    def get_d_bnd_indices( self, bnd_types ):
        uniques = set( )
        
        bnd_lst = [ ]
        for qqq, bnd in enumerate( bnd_types ):
            if bnd == "d":
                bnd_lst.append( self.diag_indicies( qqq + 1 ) )
        
        for qq in bnd_lst:
            for q in qq:
                uniques.add( q )
        
        # remove sorted?
        return sorted( list( uniques ) )
    
    def memmap( self, file ):
        return np.memmap( file,
                          dtype="float64",
                          mode="r+" if file.exists( ) else "w+",
                          shape=self.shape )
    
    def solver( self ):
        """
        Iterate through from t0 -> tn
        solve for both temp. and conc.
        """
        method = "cd"
        
        del self.T_data
        del self.C_data
        self.T_data = np.memmap( self.T_file,
                                 dtype="float64",
                                 mode="w+",
                                 shape=self.shape )
        self.C_data = np.memmap( self.C_file,
                                 dtype="float64",
                                 mode="w+",
                                 shape=self.shape )
        
        self.logg( 1, "Iterating...", )
        for i, t in enumerate( self.t ):
            # litt usikker på dimensjonene på Q
            Q = af.u_w( self.T0, self.C0, self.dh )
            
            self.ii[ 3 ] = t
            self.logg( 2, f'- t = {t}' )
            
            self.solve_next( method )
            self.T_data[ i ] = self.T1.reshape( self.shape[ 1: ] )
            self.T0, self.T1 = self.T1, np.empty( self.n )
            
            self.solve_next_C( method, Q )
            self.C_data[ i ] = self.C1.reshape( self.shape[ 1: ] )
            self.C0, self.C1 = self.C1, np.empty( self.n )
            
            self.T_data.flush( )
            self.C_data.flush( )
        
        self.logg( 1, "Finished", )
        # self.logg(1, f'Final state: {self.T0}')
    
    def logg( self, lvl, txt, logger=print ):
        """
        See config/conf.py for details
        """
        if self.logging[ lvl ]:
            logger( txt )
    
    def plot( self, t=None, x=None, y=None, z=None ):
        """
        Plot the current state
        x, y, or z: perpendicular cross-section of beef to plot.
        """
        self.plotter.show_heat_map( self.T_data, t, x, y, z )
    
    # -------------------- Retrieve data from time call --------------------
    
    def get_data_from_time( self, id: str, t: float ) -> np.array:
        '''
        Generalised retrieval of datapoint, currently supports T and C
        :param id: Either 'T' or 'C'
        :param t: time
        :return: the value of T or C at time t
        '''
        n = int( t / self.dt )
        if n < self.t.size:
            if id == 'T':
                return self.T_data[ n ]
            elif id == 'C':
                return self.C_data[ n ]
            else:
                raise ValueError( 'Trying to aquire a quantity that does not exist.' )
        else:
            raise IndexError( f'Trying to access time step no. {n} of the beef object, but it only has {self.t.size} '
                              f'entries!' )
    
    def get_T( self, t: float ) -> np.array:
        '''
        :param t: time
        :return: T at time t
        '''
        return self.get_data_from_time( self, 'T', t )
    
    def get_C( self, t: float ) -> np.array:
        '''
        :param t: time
        :return: C at time t
        '''
        return self.get_data_from_time( self, 'C', t )
    
    # ----------------------- Temperature solver -------------------------------
    
    def solve_next( self, method="cd" ):
        """
        Calculate the next time step (T1)
        """
        if method == "cd":
            A, b = self.make_Ab( )
            self.T1[ ... ] = self.T0 + \
                             (self.dt / self.a( self.ii )) * (A @ self.T0 + b)
            
            # TODO: move indexing before division
            self.T1[ self.d_bnd_indices ] = (self.gamma(
                self.ii ) / self.beta( self.ii ))[ self.d_bnd_indices ]
    
    def solve_all( self, method="cd" ):
        """
        Iterate through from t0 -> tn
        """
        # Clear temperature data before solving all
        del self.T_data
        self.T_data = np.memmap( self.T_file,
                                 dtype="float64",
                                 mode="w+",
                                 shape=self.shape )
        
        self.logg( "stage", "Iterating...", )
        for i, t in enumerate( self.t ):
            self.ii[ 3 ] = t
            self.logg( "tn", f'- t = {t}' )
            self.solve_next( method )
            self.T_data[ i ] = self.T1.reshape( self.shape[ 1: ] )
            self.T0, self.T1 = self.T1, np.empty( self.n )
            self.T_data.flush( )  # Make sure data gets written to disk
        self.logg( "stage", "Finished", )
        self.logg( "final", f'Final state: {self.T0}' )
    
    # ----------------------- Concentration solver ------------------------------
    
    def solve_next_C( self, Q, method="cd" ):
        """
        Calculate the next time step (C1)
        """
        if method == "cd":
            C, d = self.make_Cd( Q )
            self.C1[ ... ] = self.C0 + \
                             (self.dt / (2 * self.dh**2) * (C @ self.C0 + d))
    
    def solve_all_C( self, method="cd" ):
        """
        Iterate through from t0 -> tn
        """
        # Clear concentration data before solving all
        del self.C_data
        self.C_data = np.memmap( self.C_file,
                                 dtype="float64",
                                 mode="w+",
                                 shape=self.shape )
        
        self.logg( "stage", "Iterating...", )
        for i, t in enumerate( self.t ):
            self.ii[ 3 ] = t
            self.logg( "tn", f'- t = {self.t}' )
            self.solve_next_C( method )
            self.C_data[ i ] = self.C1.reshape( self.shape[ 1: ] )
            self.C0, self.C1 = self.C1, np.empty( self.n )
            self.C_data.flush( )  # Makes sure data gets written to disk
        self.logg( "stage", "Finished", )
        self.logg( "final", f'Final state: {self.C0}' )
    
    def make_Ab( self, ):
        """
        Construct A and b
        """
        # TODO: change to self.bis when other refecenres use the new implementation
        bis = self.find_border_indicies( True )
        
        # diagonal indicies
        [ k0, k1, k2, k3, k4, k5, k6 ] = self.get_ks( )
        
        # ------- contruct all diagonals -------
        
        bh2 = self.b( self.ii ) / self.dh**2
        c2h = self.c( self.ii ) / (2 * self.dh)
        
        u = self.uw( self.T0, self.C0, self.I, self.J, self.K, self.dh )
        ux = 1  # u[:, 0]
        uy = 1  # u[:, 1]
        uz = 1  # u[:, 2]
        
        C1_x = bh2 + c2h * ux
        C2_x = bh2 - c2h * ux
        C1_y = bh2 + c2h * uy
        C2_y = bh2 - c2h * uy
        C1_z = bh2 + c2h * uz
        C2_z = bh2 - c2h * uz
        
        C_u = np.array( [ C1_x, -C2_x, C1_y, -C2_y, C1_z, -C2_z ] )
        
        C3 = 6 * self.b( self.ii ) / self.dh**2
        
        _alpha = self.alpha( self.ii ).copy( )
        _alpha[ self.d_bnd_indices ] = 1  # dummy
        _alpha[ _alpha == 0 ] = 1  # dummy
        C4 = 2 * self.dh / _alpha
        
        d = np.ones( self.n )
        d0, d1, d2, d3, d4, d5, d6 = [ -C3 * d, C1_x * d,
                                       C1_y * d, C1_z * d, C2_x * d, C2_y * d, C2_z * d ]
        
        # --------------- modify the boundaries ---------------
        # see project report
        # TODO:
        # [X] set C1+C2
        # [X] set 0
        # not sure if implemented correctly, need to validate with manufactored solutions
        # - tested with neuman boundary = 0 -> behaves correctly
        # - need to validate with non-zero values / functions
        
        # add u_w to prod
        
        prod = af.dotND( bis[ :, 1: ],
                         C_u.T[ bis[ :, 0 ] ], axis=1 )  # pylint: disable=E1136
        d0[ bis[ :, 0 ] ] -= prod * C4[ bis[ :, 0 ] ] * self.beta( self.ii )[ bis[ :, 0 ] ]
        # (-self.bis[:, 1]*C2[self.bis[:, 0]] +    self.bis[:, 2]*C1[self.bis[:, 0]])*C4[self.bis[:, 0]]*self.beta(self.ii)[self.bis[:, 0]]
        
        i1, i2, i3, i4, i5, i6 = [ self.diag_indicies( i + 1 ) for i in range( 6 ) ]
        
        d1[ i1 ] = (C1_x + C2_x)[ i1 ]
        d1[ i1 + k4 ] = 0
        d1 = d1[ k1: ]
        
        d2[ i2 ] = (C1_y + C2_y)[ i2 ]
        d2[ i2 + k5 ] = 0
        d2 = d2[ k2: ]
        
        d3[ i3 ] = (C1_z + C2_z)[ i3 ]
        d3[ i3 + k6 ] = 0
        d3 = d3[ k3: ]
        
        d4[ i4 + k4 ] = (C1_x + C2_x)[ i4 + k4 ]
        d4[ i1 + k4 ] = 0
        d4 = d4[ :k4 ]
        
        d5[ i5 + k5 ] = (C1_y + C2_y)[ i5 + k5 ]
        d5[ i2 + k5 ] = 0
        d5 = d5[ :k5 ]
        
        d6[ i6 + k6 ] = (C1_z + C2_z)[ i6 + k6 ]
        d6[ i3 + k6 ] = 0
        d6 = d6[ :k6 ]
        
        # -----------------------------------------------------
        ds = [ d0, d1, d2, d3, d4, d5, d6 ]
        ks = [ k0, k1, k2, k3, k4, k5, k6 ]
        A = diags( ds, ks )
        
        b = np.zeros( self.n )
        
        prod = af.dotND( bis[ :, 1: ],
                         C_u.T[ bis[ :, 0 ] ], axis=1 )  # pylint: disable=E1136
        b[ self.bis[ :, 0 ] ] = prod * C4[ bis[ :, 0 ] ] * \
                                self.gamma( self.ii )[ bis[ :, 0 ] ]
        # (-self.bis[:, 1]*C2[self.bis[:, 0]] +  self.bis[:, 2]*C1[self.bis[:, 0]])*C4[self.bis[:, 0]]*self.gamma(self.ii)[self.bis[:, 0]]
        
        self.logg( "Ab", f'A = {A}' )
        self.logg( "Ab", f'b = {b}' )
        return A, b
    
    def make_Cd( self, Q ):
        """
        Construct C and d for Concentration equation
        """
        # diagonal indices
        [ k0, k1, k2, k3, k4, k5, k6 ] = self.get_ks( )
        ks = [ k0, k1, k2, k3, k4, k5, k6 ]
        
        # ------- construct all diagonals -------
        
        # TODO:
        # Fix \nabla u_w, currently placeholder
        # Should work
        
        D1 = 2 * self.dh * Q + const.D
        D2 = - 2 * self.dh * Q + const.D
        D3 = 6 * const.D
        
        d = np.ones( self.n )
        ds = [ D3 * d, D1[ 0 ] * d, D1[ 1 ] * d, D1[ 2 ] * d, D2[ 0 ] * d, D2[ 1 ] * d, D2[ 2 ] * d ]
        # [d0, d1, d2, d3, d4, d5, d6] = ds
        
        # TODO:
        # --------------- modify the boundaries ---------------
        
        # TODO:
        # -----------------------------------------------------
        C = diags( ds, ks )
        
        d = np.zeros( self.n )
        # ehm:
        # d[self.bis[:, 0]] = (-self.bis[:, 1] * C2 +
        #                     self.bis[:, 2] * C1) * C4 * self.gamma
        
        self.logg( "Ab", f'C = {C}' )
        self.logg( "Ab", f'd = {d}' )
        return C, d
    
    # --------------- Helper methods for make_Ab & make_Cd ---------------
    
    def index_of( self, i, j, k ):
        """
        Returns the 1D index from 3D coordinates
        """
        return i + self.I * j + self.I * self.J * k
    
    # Unnecessary?
    def index_of_inverse( self, p ):
        """
        Returns the 3D index from 1D coordinate
        """
        k = p // (self.I * self.J)
        j = (p - k * self.I * self.J) // self.I
        i = (p - k * self.I * self.J - j * self.I)
        return i, j, k
    
    def get_ks( self ):
        """
        Get the ks to use in diags(ds, ks) in self.make_Ab
        """
        k0 = 0
        k1 = self.index_of( 1, 0, 0 )
        k2 = self.index_of( 0, 1, 0 )
        k3 = self.index_of( 0, 0, 1 )
        k4 = self.index_of( -1, 0, 0 )
        k5 = self.index_of( 0, -1, 0 )
        k6 = self.index_of( 0, 0, -1 )
        return k0, k1, k2, k3, k4, k5, k6
    
    def diag_indicies( self, bnd ):
        """
        Finds the indices of a specific boundary

        bnd:
        - 1: x = 0
        - 2: y = 0
        - 3: z = 0
        - 4: x = X
        - 5: y = Y
        - 6: z = Z

        Used to index the diagonals

        May only be run one time for each diagonal
        """
        
        i = (bnd == 4 and [ self.I - 1 ]) or (bnd == 1 and [ 0 ]) or range( self.I )
        j = (bnd == 5 and [ self.J - 1 ]) or (bnd == 2 and [ 0 ]) or range( self.J )
        k = (bnd == 6 and [ self.K - 1 ]) or (bnd == 3 and [ 0 ]) or range( self.K )
        # just ignore sort? it doesn't break without it
        return np.sort( np.array( self.index_of( *np.meshgrid( i, j, k ) ) ).T.reshape( -1 ) )
    
    def find_border_indicies( self, new=False ):
        """
        returns the indices for every boundary node

        [[index, x0, xn, y0, yn, z0, zn],...]

        E.g: (0,Y,2) \n
        [[index, 1, 0, 0, 1, 0, 0],...]

        TODO: remove old implementation
        """
        if new:
            indicies = np.zeros( (self.border, 7), dtype=np.int16 )
            tmp = 0
            for k in range( self.K ):
                for j in range( self.J ):
                    for i in range( self.I ):
                        if i == 0 or i == (self.I - 1) or j == 0 or j == (self.J - 1) or k == 0 or k == (self.K - 1):
                            indicies[ tmp ] = np.array(
                                [ self.index_of( i, j, k ), *self.sum_start_and_end( i, j, k, True ) ] )
                            tmp += 1
            return indicies
        else:
            indicies = np.zeros( (self.border, 3), dtype=np.int16 )
            tmp = 0
            for k in range( self.K ):
                for j in range( self.J ):
                    for i in range( self.I ):
                        if i == 0 or i == (self.I - 1) or j == 0 or j == (self.J - 1) or k == 0 or k == (self.K - 1):
                            indicies[ tmp ] = np.array(
                                [ self.index_of( i, j, k ), *self.sum_start_and_end( i, j, k ) ] )
                            tmp += 1
            return indicies
    
    def sum_start_and_end( self, i, j, k, new=False ):
        """
        wacky way to count the different boundaries the node borders

        E.g: \n
        (0, 0, 0) -> [3, 0] \n
        (0, 0, Z) -> [2, 1] \n
        (0, 4, Z) -> [1, 1]
        TODO:  remove old implementation
        """
        if new:
            x0 = int( i == 0 )
            xn = int( i == self.I - 1 )
            y0 = int( j == 0 )
            yn = int( j == self.J - 1 )
            z0 = int( k == 0 )
            zn = int( k == self.K - 1 )
            return x0, xn, y0, yn, z0, zn
        else:
            # pretend you didn't see this
            boundaries = ((i == 0 and "start") or (i == self.I - 1 and "end"),
                          (j == 0 and "start") or (j == self.J - 1 and "end"),
                          (k == 0 and "start") or (k == self.K - 1 and "end"))
            
            start = 0
            end = 0
            for boundary in boundaries:
                start += 1 if boundary == "start" else 0
                end += 1 if boundary == "end" else 0
            return start, end
    
    # --------------- Misc. -------------------------------------------
    
    def pre_check( self, conf, T_conf, C_conf ):
        def _check_T_or_C( conf, prefix ):
            assert conf[ "pde" ][ "a" ] is not None, f'{prefix}: a should not be None'
            assert conf[ "pde" ][ "b" ] is not None, f'{prefix}: b should not be None'
            assert conf[ "pde" ][ "c" ] is not None, f'{prefix}: c should not be None'
            assert conf[ "bnd" ][ "alpha" ] is not None, f'{prefix}: alpha should not be None'
            assert conf[ "bnd" ][ "beta" ] is not None, f'{prefix}: beta should not be None'
            assert conf[ "bnd" ][ "gamma" ] is not None, f'{prefix}: gamma should not be None'
            assert conf[ "uw" ] is not None, f'{prefix}: uw should not be None'
            assert 0 <= len( conf[ "bnd_types" ] ) <= 6, \
                f'{prefix}: bnd_types should be of length 0-6, got length={len( conf[ "bnd_types" ] )}'
            assert conf[ "initial" ] is not None, f'{prefix}: initial should not be None'
        
        assert conf[ "dh" ] > 0, f'conf: dh should be >0, got dh={conf[ "dh" ]}'
        assert conf[ "dt" ] > 0, f'conf: dt should be >0, got dt={conf[ "dt" ]}'
        assert conf[ "tlen" ] > 0, f'conf: tlen should be >0, got tlen={conf[ "tlen" ]}'
        assert conf[ "tn" ] > conf[ "t0" ], \
            f'conf: tn should be >t0, got t0={conf[ "t0" ]} and tn={conf[ "tn" ]}'
        
        dims = conf[ "dims" ]
        assert dims[ "xlen" ] > 0, f'conf: xlen should be >0, got xlen={dims[ "xlen" ]}'
        assert dims[ "xn" ] > dims[ "x0" ], \
            f'conf: xn should be >x0, got x0={dims[ "x0" ]} and xn={dims[ "xn" ]}'
        assert dims[ "ylen" ] > 0, f'conf: ylen should be >0, got ylen={dims[ "ylen" ]}'
        assert dims[ "yn" ] > dims[ "y0" ], \
            f'conf: yn should be >y0, got y0={dims[ "y0" ]} and yn={dims[ "yn" ]}'
        assert dims[ "zlen" ] > 0, f'conf: zlen should be >0, got zlen={dims[ "zlen" ]}'
        assert dims[ "zn" ] > dims[ "z0" ], \
            f'conf: zn should be >z0, got z0={dims[ "z0" ]} and zn={dims[ "zn" ]}'
        
        _check_T_or_C( T_conf, "T" )
        _check_T_or_C( C_conf, "C" )
    
    def __str__( self ):
        return "Sorry can't do... yet"
