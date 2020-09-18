import numpy as np
from dataset_constructor import datasetConstructor
import matplotlib.pyplot as plt


n_components = [i for i in range(15)]
y = [[], []]
n_iterations = 35
for i in n_components:
	print(i)
	# Set a seed
	np.random.seed(15)

	# Generate a dataset with the distribution given by 'method'
	constructor = datasetConstructor(n_samples=1000, n_components=i, method='gaussian')
	dataset = constructor.dataset()
	[y[j].append(constructor.time[j]) for j in range(2)]


# Plot training errors
x = n_components
plt.plot(x, y[0], linewidth=0.8)
plt.plot(x, y[1], linewidth=0.8)
plt.title("Construction methods times")
plt.xlabel("# Components")
plt.ylabel("Time (s)")
plt.xticks([i for i in x if i % 2 == 0])
plt.legend(["'Brute force' method", "'Binary search' method"])
plt.grid(color='#efefef')
plt.show()
