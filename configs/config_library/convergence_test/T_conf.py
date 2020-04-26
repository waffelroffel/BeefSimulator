import numpy as np
from auxillary_functions import u_w

# alp = 0.125e-6
alp = 1e-4
Lx = 0.075
Ly = 0.039
Lz = 0.054
# alp = alpha = heat transfer coeff. This produces the eq. dT/dt = alpha*grad^2(T)


def T(xx, yy, zz, t):
    return 3 * np.exp(-4*alp*(np.pi)**2*(1/Lx+1/Ly+4/Lz) * t) * \
        np.sin(2*np.pi/Lx * xx) * np.sin(2*np.pi/Ly * yy) * np.sin(4*np.pi/Lz * zz)


def T_initial(T, C, shape, xx, yy, zz, t):
    return 3 * np.sin(2*np.pi/Lx * xx) * np.sin(2*np.pi/Ly * yy) * np.sin(4*np.pi/Lz * zz)


T_a = 1

T_b = alp

T_c = 0

'''
Diff eq has Neumann boundary conditions

dT/dx|_0  = f_x0
dT/dx|_Lx = f_xL
dT/dy|_0  = f_y0
dT/dy|_Ly = f_yL
dT/dz|_0  = f_z0
dT/dz|_Lz = f_zL
T(x,y,z,0) =  3*sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*sin(4*pi*z/Lz)
'''

xi = lambda x: np.sin( 2 * np.pi / Lx * x )
eta = lambda y: np.sin( 2 * np.pi / Ly * y )
zeta = lambda z: np.sin( 4 * np.pi / Lz * z )
tau = lambda t: np.exp( -4 * alp * t * np.pi**2 * (1 / Lx**2 + 1 / Ly**2 + 4 / Lz**2) )

f_x0 = lambda y, z, t: 2 * np.pi / Lx * eta( y ) * zeta( z ) * tau( t )
f_xL = lambda y, z, t: 2 * np.pi / Lx * eta( y ) * zeta( z ) * tau( t )
f_y0 = lambda x, z, t: xi( x ) * 2 * np.pi / Ly * zeta( z ) * tau( t )
f_yL = lambda x, z, t: xi( x ) * 2 * np.pi / Ly * zeta( z ) * tau( t )
f_z0 = lambda x, y, t: xi( x ) * eta( y ) * 4 * np.pi / Lz * tau( t )
f_zL = lambda x, y, t: xi( x ) * eta( y ) * 4 * np.pi / Lz * tau( t )


def T_alpha(T, C, shape, xx, yy, zz, t):
    return 1


def T_beta(T, C, shape, xx, yy, zz, t):
    return 0


def T_gamma(T, C, shape, xx, yy, zz, t):
    temp = np.zeros(shape)
    temp[0,:,:] = f_x0(yy, zz, t)[0,:,:]
    temp[-1,:,:] = f_xL(yy, zz, t)[-1,:,:]
    temp[:,0,:] = f_y0(xx, zz, t)[:,0,:]
    temp[:,-1,:] = f_yL(xx, zz, t)[:,-1,:]
    temp[:,:,0] = f_z0(xx, yy, t)[:,:,0]
    temp[:,:,-1] = f_zL(xx,yy,t)[:,:,-1]
    return temp.flatten()


def uw(T, C, I, J, K, dh):
    u = u_w(T.reshape((I, J, K)), C.reshape((I, J, K)), dh)
    return u.reshape((3, -1)).T


# T_bnd_types = ["d", "d", "d", "d", "d", "d"]
T_bnd_types = []

T_conf = {
    "pde": {"a": T_a,
            "b": T_b,
            "c": T_c},
    "bnd": {"alpha": T_alpha,
            "beta": T_beta,
            "gamma": T_gamma},
    "uw": uw,
    "bnd_types": T_bnd_types,
    "initial": T_initial,
    "analytic": T
}
