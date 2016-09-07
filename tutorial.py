import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

#Set the integer starting value used to generate random numbers
np.random.seed(5)

#Create a dataset that looks like two interleaved half circles. The first parameter is the number of data points to generate, the second is the Gaussian noise level
x, y = datasets.make_moons(300, noise=0.20)

#Plot the dataset so we can see what has been generated
plt.scatter(x[:,0], x[:,1], s=40, c=y, cmap=plt.cm.Spectral)

plt.show()
