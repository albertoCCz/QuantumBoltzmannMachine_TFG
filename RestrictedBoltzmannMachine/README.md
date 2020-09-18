## Restricted Boltzmann Machine

In this section we make use of the [Erik Linder-Norén's code](https://github.com/eriklindernoren/ML-From-Scratch) for the Restricted Boltzmann Machine in order to understand in a practical way its functionig. In order to use this model, we developt a method to generate a dataset of binary vectors which follow a particular distribution, being the possibilities a gaussian/normal distribution, a exponential distribution or a uniform distribution.

After doing this, we study briefly the next few points:
- The time it takes for each construction dataset method ('Brute force' and 'Binary search' methods) to construct the dataset based on the number of components of the binary vectors.
- How the model error behaves depending on the number of samples in the dataset, if we fix the length of the binary vectors.
- How the model error behaves depending on the distribution with which we generated the dataset.

The results indicate the following:
- The best method to use in order to generate the dataset is the one we called 'Binary search' method.

![Construction methods times](https://github.com/albertoCCz/QuantumBoltzmannMachine_TFG/blob/master/RestrictedBoltzmannMachine/Studies/Images/Construction_methods_times.png)

- With a fixed length for the vectors, the biggest the number of samples we have the best they follow the selected distribution and the best the model is capable of reproduce this distribution.

![Error depending on the number of samples](https://github.com/albertoCCz/QuantumBoltzmannMachine_TFG/blob/master/RestrictedBoltzmannMachine/Studies/Images/RBM_training.png)

- It seems that the gaussian distribution is the only one that allowed the model to improve during the simulation. In the case of the uniform distribution the reason why it can not improve is obvious, but not in the case of the exponential distribution. An explanation for this would requires of further research.

![Error depending on the distribution](https://github.com/albertoCCz/QuantumBoltzmannMachine_TFG/blob/master/RestrictedBoltzmannMachine/Studies/Images/Distribution_methods.png)
