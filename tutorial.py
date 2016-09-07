import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

#Define the number of data points we're working with
num_points = 300

#Number of input layer dimensions
num_input_dim = 2

#Number of output layer dimensions
num_output_dim = 2 

#Learning rate for gradient descent - from tutorial
epsilon = 0.01

#Regularization strength - from tutorial
reg_lambda = 0.01

#Helper function to evaluate the total loss on the dataset
def calculate_loss(model, x, y):

    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']

    #Forward propagation to calculate our predictions
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    #Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)

    #Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))

    return 1./num_examples * data_loss

#Helper function to predict an output (0 or 1)
def predict(model, x):

    w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']

    #Forward propagation
    z1 = x.dot(w1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(w2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)

def main():

	#Set the integer starting value used to generate random numbers
	np.random.seed(5)

	#Create a dataset that looks like two interleaved half circles. The first parameter is the number of data points to generate, the second 		is the Gaussian noise level
	x, y = datasets.make_moons(num_points, noise=0.20)






