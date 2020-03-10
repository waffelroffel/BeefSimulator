import numpy as np

'''
Make data for testing of plot etc.

With a pixel size of f.ex. 0.001 meters, if we assume the beef slice to have dimensions 10x15x20 cm
Then the dataset has dimensions 100*150*200

Import this file and find test data in a,b = make_testdata() where a, b are orthorhombic 3D numpy arrays

'''

def make_testdata() -> (np.array, np.array):
	shape = (100, 150, 200)
	data = np.ones(shape)*50  # Set all values to 50 (degrees?)
	
	# Sine variation in z-direction from 25 to 75, constant in xy-plane
	variation = np.zeros_like(data)
	for j in range(200):
		variation[:, :, j] = 25*np.sin(2*np.pi*j/200)
	data1 = data + variation
	
	#Hot 'bands' in the xy-plane, constant over z
	variation = np.zeros_like(data)
	y_indices = np.zeros(100)
	for j in range(100):
		for k in np.arange(-25, 25):
			variation[j, 75 + int(50*np.sin(j/6)) + k, :] = 25 - abs(k)
	data2 = data + variation
	
	return data1, data2

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

