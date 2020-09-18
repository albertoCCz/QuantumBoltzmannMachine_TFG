import numpy as np
from dataset_constructor import datasetConstructor
from restricted_boltzmann_machine import RBM
import matplotlib.pyplot as plt


samples = [1000, 10000, 100000]
y = []
n_iterations = 50
for i in samples:
	# Set a seed
	np.random.seed(15)

	# Generate a dataset with the distribution given by 'method'
	constructor = datasetConstructor(n_samples=i, n_components=10, method='gaussian')
	dataset = constructor.dataset()

	# Train the Restricted Boltzmann Machine on the generated data
	rbm = RBM(n_hidden=4, learning_rate=0.1, batch_size=10, n_iterations=n_iterations)
	fit = rbm.fit(dataset)

	y.append(rbm.training_errors)


# Plot training errors
x = np.arange(0, n_iterations, 1)
plt.plot(x, y[0], linewidth=0.8)
plt.plot(x, y[1], linewidth=0.8)
plt.plot(x, y[2], linewidth=0.8)
plt.title("RMB training errors")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.legend([str(i) + " samples" for i in samples])
plt.grid(color='#efefef')
plt.show()
