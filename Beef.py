import numpy as np

class Beef:
	def __init__(self, size: np.array = np.array([1,1,1]), h: float = 0.1):
		pass
	
	def load_from_file(self, filename: str):
		pass
	
	T = [[[]]] # = Temperature dist.
	C = [[[]]] # = Moisture concentration dist.
	shape = T.shape
	
	
	def get_T(self):
		return self.T
	def get_C(self):
		return self.C
	
	
	