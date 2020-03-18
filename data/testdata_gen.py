import numpy as np

'''
Make data for testing of plot etc.

With a pixel size of f.ex. 0.001 meters, if we assume the beef slice to have dimensions 10x15x20 cm
Then the dataset has dimensions 100*150*200

Import this file and find test data in a,b = make_testdata() where a, b are orthorhombic 3D numpy arrays

'''


def make_testdata(shape) -> (np.array, np.array):
    data = np.ones(shape)*50  # Set all values to 50 (degrees?)

    # Sine variation in z-direction from 25 to 75, constant in xy-plane
    variation = np.zeros_like(data)
    for j in range(shape[2]):
        variation[:, :, j] = 25*np.sin(2*np.pi*j/shape[2])
    data1 = data + variation

    # Hot 'bands' in the xy-plane, constant over z
    variation = np.zeros_like(data)
    for j in range(shape[0]):
        for k in np.arange(-25, 25):
            variation[j, 75 + int(50*np.sin(j/6)) + k, :] = 25 - abs(k)
    data2 = data + variation

    return data1, data2


def make_manisol_1d(xd: int, td: int) -> np.array:
    '''
    Boundary value problem:
    Solve T_t = a * T_x^2 (where f_q means partial derivative of f with respect to q) for x in [0,L], t in [0, inf)
    (1): T(0, t) = 0
    (2): T(L, t) = 0
    (3): T(x, 0) = sin(2*pi*x/L)		(Chosen for convenience)
    Only the n=2-Fourier component is non-zero due to B.C. (3) => T(x,t) = exp(-4*a*pi^2/L^2 * t) * sin(2*pi/L * x)
    Insert a = 1e-3, L = 1

    :param xd: number of points discretising x
    :param td: number of points discretising t
    :return: The manifactured solution to the BVP described above
    '''
    # Analytic solution
    def T(x, t): return np.exp(-4*1e-3*(np.pi)**2 * t) * np.sin(2*np.pi * x)

    x = np.linspace(0, 1, xd)
    delta_t = 0.1
    t = np.linspace(0, td*delta_t, td)
    xx, tt = np.meshgrid(x, t)

    return T(xx, tt)


def make_manisol_3d(shape_d: tuple, td: int) -> np.array:
    '''
    Boundary value problem:
    Solve T_t = a * (T_x^2 + T_y^2 + T_z^2) for x in [0, Lx], y in [0, Ly], z in [0, Lz], t in [0, inf)
    (1.1): T(0,y,z,t) = 0		(1.2): T(Lx,y,z,t) = 0
    (2.1): T(x,0,z,t) = 0		(2.2): T(x,Ly,z,t) = 0
    (3.1): T(x,y,0,t) = T(x,y,Lz,t)			(3.2): T_z(x,y,0,t) = T_z(x,y,Lz,t)
    (4): T(x,y,z,0) = 3*sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*sin(4*pi*z/Lz)			(Chosen for convenience)
    Only the (nx=2, ny=2, nz=2)-Fourier component is non-zero due to B.C. (4) => ...
    T(x,y,z,t) = 3 * exp(-4*a*pi^2*(1/Lx^2+1/Ly^2+4/Lz^2) * t) * sin(2*pi/Lx * x) * sin(2*pi/Ly * y) * sin(4*pi/Lz * z)
    insert a = 1e-3, Lx=Ly=Lz=1

    ### NB! (3.1) -> T(x,y,0,t) = 0 and (3.2) -> T(x,y,Lz,t) = 0 gives exactly the same solution!

    :param shape_d: tuple[int] of the numbers of points discretising x, y and z (Nx, Ny, Nz)
    :param td: number of points discretising t
    :return: The manifactured solution to the BVP described above
    '''
    # Analytic solution
    def T(x, y, z, t): return 3 * np.exp(-4*1e-3*(np.pi)**2*(1+1+4) * t) * \
        np.sin(2*np.pi * x) * np.sin(2*np.pi * y) * np.sin(4*np.pi * z)

    xd, yd, zd = shape_d
    x = np.linspace(0, 1, xd)
    y = np.linspace(0, 1, yd)
    z = np.linspace(0, 1, zd)
    delta_t = 0.1
    t = np.linspace(0, td*delta_t, td)
    xx, yy, zz, tt = np.meshgrid(x, y, z, t)
    return T(xx, yy, zz, tt)


def test_diff_manisol(calcdata: np.array, manidata: np.array) -> float:
    '''

    :param calcdata: Calculated data
    :param manidata: Manifactured data
    :return: (Frobenius) matrix norm of the absolute value of the difference
    (Note: This squelches all imaginary parts too)
    '''
    return np.linalg.norm(np.abs(calcdata - manidata))
    # TODO: Denne bør også returnere punktvis feil

# TODO: Dette kan abstraheres og automatiseres enda mer hvis lesing og skriving til fil er på plass. Lag i så fall
# TODO: ... en funksjon som henter inn filnavn, bruker innlesing og kaller test_diff_manisol på dem
# TODO: ... Man kan evt. legge til produksjon av disse datasett og lagre dem i __main__ under.		--Svein


'''
Very simple test data generator.
'''
if __name__ == '__main__':
    h = 0.25
    dt = 0.1

    shape = (101, 151, 201)
    t = np.arange(0, 10.1, dt)
    x = np.linspace(0, 25.00, shape[0])
    y = np.linspace(0, 37.50, shape[1])
    z = np.linspace(0, 50.00, shape[2])

    d_shape = (len(t), len(x), len(y), len(z))
    print(d_shape)

    U = np.random.random(d_shape)
    C = np.random.random(d_shape)

    U[:, :, :, 0] = 0.00
    U[:, :, 0, :] = 25.0
    U[:, :, :, -1] = 50.0
    U[:, :, -1, :] = 75.0
    U[:, 0, :, :] = 100.0
    U[:, -1, :, :] = 125.0
    U[0, :, :, :], U[10, :, :, :] = make_testdata(shape)

    C[:, :, :, 0] = 0.00
    C[:, :, 0, :] = 0.25
    C[:, :, :, -1] = 0.50
    C[:, :, -1, :] = 0.75
    C[:, 0, :, :] = 1.00
    C[:, -1, :, :] = 1.25

    np.save('test_temp_dist.npy', U)
    np.save('test_cons_dist.npy', C)
