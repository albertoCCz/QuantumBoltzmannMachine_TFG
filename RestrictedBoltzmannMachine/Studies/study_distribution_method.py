import numpy as np
from dataset_constructor import datasetConstructor
from restricted_boltzmann_machine import RBM
import matplotlib.pyplot as plt


method = ["gaussian", "exponential", "uniform"]
y = []
n_iterations = 35
for i in method:
	# Set a seed
	np.random.seed(15)

	# Generate a dataset with the distribution given by 'method'
	constructor = datasetConstructor(n_samples=10000, n_components=3, method=i)
	dataset = constructor.dataset()

	# Train the Restricted Boltzmann Machine on the generated data
	rbm = RBM(n_hidden=4, learning_rate=0.1, batch_size=50, n_iterations=35)
	fit = rbm.fit(dataset)

	y.append(rbm.training_errors)


# Plot training errors
x = np.arange(0, n_iterations, 1)
plt.plot(x, y[0], linewidth=0.8)
plt.plot(x, y[1], linewidth=0.8)
plt.plot(x, y[2], linewidth=0.8)
plt.title("RMB training errors for each distribution")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.legend([i for i in method])
plt.grid(color='#efefef')
plt.show()
