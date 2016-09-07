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
    corect_logprobs = -np.log(probs[range(num_points), y])
    data_loss = np.sum(corect_logprobs)

    #Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))

    return 1./num_points * data_loss

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

#This function learns parameters for the neural network and returns the model.
#num_hidden - Number of nodes in the hidden layer
#num_passes - Number of passes through the training data for gradient descent
#print_loss - If True, print the loss every 1000 iterations
def build_model(x, y, num_hidden, num_passes=20000, print_loss=False):
     
    #Initialize the parameters to random values. We need to learn these.
    np.random.seed(5)

    w1 = np.random.randn(num_input_dim, num_hidden) / np.sqrt(num_input_dim)
    b1 = np.zeros((1, num_hidden))

    w2 = np.random.randn(num_hidden, num_output_dim) / np.sqrt(num_hidden)
    b2 = np.zeros((1, num_output_dim))
 
    #This is what we return at the end
    model = {}
     
    #Gradient descent. For each batch...
    for i in xrange(0, num_passes):
 
        # Forward propagation
        z1 = x.dot(w1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(w2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        #Backpropagation
        delta3 = probs
        delta3[range(num_points), y] -= 1
        dw2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(w2.T) * (1 - np.power(a1, 2))
        dw1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        #Add regularization terms (b1 and b2 don't have regularization terms)
        dw2 += reg_lambda * w2
        dw1 += reg_lambda * w1
 
        #Gradient descent parameter update
        w1 += -epsilon * dw1
        b1 += -epsilon * db1
        w2 += -epsilon * dw2
        b2 += -epsilon * db2
         
        #Assign new parameters to the model
        model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
         
        #Optionally print the loss.
        #This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, calculate_loss(model, x, y))
     
    return model

def plot_decision_boundary(pred_func, x, y):

    #Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01

    #Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    #Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.PiYG)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.PiYG)
    plt.show()

def main():

	#Set the integer starting value used to generate random numbers
	np.random.seed(5)

	#Create a dataset that looks like two interleaved half circles. The first parameter is the number of data points to generate, the second 		is the Gaussian noise level
	#x, y = datasets.make_blobs(num_points, centers = [(3, 3), (6, 6), (0, 0)])
	x, y = datasets.make_moons(num_points, noise=0.2)

	plt.scatter(x[:,0], x[:,1], s=50, c=y, cmap=plt.cm.PiYG)
	plt.show()
	
	#Build the model
	model = build_model(x, y, 3, print_loss=True)

	#Plot the result
	plot_decision_boundary(lambda x:predict(model, x), x, y)

if __name__ == "__main__":
    main()






