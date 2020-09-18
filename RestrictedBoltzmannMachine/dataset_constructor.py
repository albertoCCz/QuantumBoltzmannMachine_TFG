import numpy as np
from numpy.random import default_rng
import time


rng = default_rng()


class datasetConstructor():
	"""
	Constructor parameters:
	------------------------
	n_samples:
		int Number of samples to generate for the dataset
	n_components:
		int Number of components/dimension of each sample
	method:
		string Probability distribution to sample from. Possibilities are:
		'gaussian', 'uniform'
	"""

	def __init__(self, n_samples=100, n_components=10, method='gaussian'):
		self.n_samples = n_samples
		self.n_components = n_components
		self.method = method
		self.time = []

	def dataset(self):
		size = self.n_samples
		# Select distribution constructor
		if self.method == 'gaussian':
			dec_set = rng.normal(loc=0.5, scale=0.25, size=size)
			dec_set = self._to01(dec_set)
		elif self.method == 'exponential':
			dec_set = rng.exponential(scale=1/5, size=size)
			dec_set = self._to01(dec_set)
		elif self.method == 'uniform':
			dec_set = rng.uniform(size=size)
			dec_set = self._to01(dec_set)
		else:
			return "No valid method."

		n_config = 2 ** self.n_components

		## First method: 'Brute force' method
		start_1 = time.time()
		# Generate all possible binary states with n_components bits
		# dec_states = [i for i in range(n_config)]
		# zeros_add = [[0 for j in range(self.n_components - len(bin(dec_states[i])[2:]))] for i in dec_states]
		# bin_states = zeros_add.copy()
		# [bin_states[i].extend([int(j) for j in bin(dec_states[i])[2:]]) for i in range(n_config)]
		#
		# # Transform dec_set into bin_set
		# limits = [i/n_config for i in range(n_config + 1)]
		# bin_set = []
		# for i in dec_set:
		# 	if i == 0:
		# 		bin_set.append(bin_states[0])
		# 	for j in range(len(limits)-1):
		# 		low = limits[j]
		# 		high = limits[j+1]
		# 		if low < i <= high:
		# 			bin_set.append(bin_states[j])

		end_1 = time.time()

		## Second method: 'Binary search' method
		start_2 = time.time()
		# Generate intervals
		intervals = []
		for i in range(len(dec_set)):
			low = 0
			high = 1 / 2
			if dec_set[i] == 0:
				intervals.extend([0])
			else:
				for j in range(self.n_components):
					if low < dec_set[i] <= high:
						high = low + (high - low) / 2
					else:
						distance = high - low
						low = high
						high = low + distance / 2

				intervals.extend([int(low * n_config)])

		# Generate binary states
		z_add = [[0 for j in range(self.n_components - len(bin(i)[2:]))] for i in intervals]
		[z_add[i].extend([int(j) for j in bin(intervals[i])[2:]]) for i in range(self.n_samples)]
		bin_states_2 = z_add.copy()
		end_2 = time.time()

		self.time = [end_1 - start_1, end_2 - start_2]

		return np.asarray(bin_states_2)

	def _to01(self, X):
		X = X - min(X)
		return X/max(X)
