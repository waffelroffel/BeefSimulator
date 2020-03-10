'''
Make data for testing of plot etc.

With a pixel size of f.ex. 0.001 meters, if we assume the beef slice to have dimensions 10x15x20 cm
Then the dataset has dimensions 100*150*200

Import this file and find test data in a,b = make_testdata() where a, b are orthorhombic 3D numpy arrays

'''
import numpy as np


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

