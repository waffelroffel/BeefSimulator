import numpy as np
'''
Very simple test data generator.
'''
if __name__ == '__main__':
    h = 0.25
    dt = 0.1

    t = np.arange(0, 10.1, dt)
    x = np.arange(0, 5.25, h)
    y = np.arange(0, 4.25, h)
    z = np.arange(0, 3.25, h)

    d_shape = (len(t), len(x), len(y), len(z))

    U = np.random.random(d_shape)
    C = np.random.random(d_shape)

    U[:,:,:, 0] = 0.00
    U[:,:, 0,:] = 0.25
    U[:,:,:, -1] = 0.50
    U[:,:, -1,:] = 0.75
    U[:, 0,:,:] = 1.00
    U[:, -1,:,:] = 1.25

    C[:,:,:, 0] = 0.00
    C[:,:, 0,:] = 0.25
    C[:,:,:, -1] = 0.50
    C[:,:, -1,:] = 0.75
    C[:, 0,:,:] = 1.00
    C[:, -1,:,:] = 1.25

    U = np.einsum('n, nijk -> nijk', np.arange(0, 101), U)
    C = np.einsum('n, nijk -> nijk', np.arange(0, 10.1, 0.1), U)

    np.save('test_temp_dist.npy', U)
    np.save('test_cons_dist.npy', C)