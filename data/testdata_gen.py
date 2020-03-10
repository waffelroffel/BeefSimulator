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
	
	#Hot 'bands' in the xy-plane, constant over z
	variation = np.zeros_like(data)
	for j in range(shape[0]):
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

    shape = (101, 151, 201)
    t = np.arange(0, 10.1, dt)
    x = np.linspace(0, 25.00, shape[0])
    y = np.linspace(0, 37.50, shape[1])
    z = np.linspace(0, 50.00, shape[2])

    d_shape = (len(t), len(x), len(y), len(z))
    print(d_shape)

    U = np.random.random(d_shape)
    C = np.random.random(d_shape)

    U[:,:,:, 0] = 0.00
    U[:,:, 0,:] = 25.0
    U[:,:,:, -1] = 50.0
    U[:,:, -1,:] = 75.0
    U[:, 0,:,:] = 100.0
    U[:, -1,:,:] = 125.0
    U[0,:,:,:], U[10,:,:,:] = make_testdata(shape)

    C[:,:,:, 0] = 0.00
    C[:,:, 0,:] = 0.25
    C[:,:,:, -1] = 0.50
    C[:,:, -1,:] = 0.75
    C[:, 0,:,:] = 1.00
    C[:, -1,:,:] = 1.25

    np.save('test_temp_dist.npy', U)
    np.save('test_cons_dist.npy', C)

