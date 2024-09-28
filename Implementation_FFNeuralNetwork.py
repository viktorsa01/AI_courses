import numpy as np
from pathlib import Path
from typing import Tuple


def func(X: np.ndarray) -> np.ndarray:
    """
    The data generating function.
    Do not modify this function.
    """
    return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def noisy_func(X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
    """
    Add Gaussian noise to the data generating function.
    Do not modify this function.
    """
    return func(X) + np.random.randn(len(X)) * epsilon


def get_data(n_train: int, n_test: int):
    """
    Generating training and test data for
    training and testing the neural network.
    Do not modify this function.
    """
    X_train = np.random.rand(n_train, 2) * 2 - 1
    y_train = noisy_func(X_train)
    X_test = np.random.rand(n_test, 2) * 2 - 1
    y_test = noisy_func(X_test)

    return X_train, y_train, X_test, y_test

def linear_activation(x):
    return x

def sigmoid_activation(x):
    return 1/(1 + np.exp(-x))

def MSE (y_hat, y):
    return np.mean((y_hat- y)**2)

def d_sigmoid(x):
    #derivative of the sigmoid function
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

def d_linear(x):
    return 1

class Node:
    def __init__(self, num_of_weights) :
        '''
        input:
            num_of_weights: integer specifying the number of wights into this node
        '''
        self.weights = np.array(np.zeros(num_of_weights)) + 0.01

class Layer:
    def __init__(self, size, size_prev_layer, intercept, activation_function):
        '''
        input:
            size: int specifying number of nodes in the layer
            size_prev_layer: int specifying the number of nodes in the previous layer
            intercept: boolean specifying wether the layer has an intercept or not
            activation_function: activation function to use at this layer, takes a real number as input and returns a real number
        '''
        self.size = size
        self.intercept = intercept
        #List of all nodes in the layer
        self.nodes = []
        #Add potential intercept node
        if intercept:
            self.nodes.append(Node(0))
        for i in range (size-len(self.nodes)):
            self.nodes.append(Node(size_prev_layer))
        
        self.activation_function = activation_function
    
    def activation(self, x):
        '''
        input:
            x: numpy array of real numbers
        return:
            numpy array of activations of each node in the layer
        '''
        #I feel like there might be a more efficient way to implement this, but dont see how...
        
        activation = np.zeros(self.size)
        
        if self.intercept:
            activation[0] = 1
        #Check if this layer har no previous layer, if so return activation of the input plus potetial intercept
        if self.nodes[self.intercept].weights.size == 0:
            activation = self.activation_function(x)
            if self.intercept:
                activation = np.insert(x, 0, 1)
            return activation
        
        #calculate activation of all non-intercept nodes
        for i in range(self.intercept, self.size):
            activation[i] = self.activation_function(( np.dot(self.nodes[i].weights, x) ))
        
        return activation
        

class Feedforward_NN:
    def __init__(self, nodes_in_layers, intercept, activation_func ):
        '''
        Input: 
            layers: list of integers specifying the number of nodes in each layer (not counting the intercept)
            intercept: list of boolean specifying wether the layer has an intercept node
            activation_func: list of function specifying the activation function for each layer
            Note all of the above should be the same length as the number of layers in the network
        '''
        #Make a list of lists, representing each layer in the network, with numpy arrays representing the wiegths into the specific node
        self.layers = [Layer(nodes_in_layers[0], 0, intercept[0], activation_func[0])]
        for i in range(1,len(nodes_in_layers)):
            self.layers.append(Layer(nodes_in_layers[i], nodes_in_layers[i-1], intercept[i], activation_func[i]))
        
    def inference(self, x):
        '''
        input:
            x: numpy array of real numbers 
        return:
            predicted response from the NN
        '''
        for i in range(len(self.layers)):
            x = self.layers[i].activation(x)
        return x
    
    def activation_nodes(self, x):
        '''
        input:
            x: numpy array of real numbers 
        return:
            list of numpy arrays containing activation of all nodes in the network
        '''
        act = []
        for i in range(len(self.layers)):
            x = self.layers[i].activation(x)
            act.append(x)
        return act
        
    def train_NN(self, x_train, y_train, h, derivatives):
        '''
        input:
            x_train: numpy array of real valued, inputs into the neural network 
            y_train: numpy array of real valued responses
            Note x_train and y_train should contain equal number of rows(samples)
            h: float stepsize
            derivatives: list cotaining derviatives of activation functions in each layer
        '''
        #Train the network using all samples from the training set
        for sample in range(len(x_train)):
            delta = []
            a = self.activation_nodes(x_train[sample])
            #Calculate the delta_h from the output nodes and the hidden nodes
            d = derivatives[-1](a[-1]) * (y_train[sample] - a[-1])
            delta.append( d )
            
            for i_lay in range(len(self.layers)-2, 0, -1):
                layer = self.layers[i_lay]
                d = []
                for i_node in range(layer.intercept, len(layer.nodes)):
                    sum_term = 0
                    #probably a way more efficient way to do this but I think I would have to restructure a lot of my code :(
                    for j in range(self.layers[i_lay + 1].intercept, len(self.layers[i_lay + 1].nodes)):
                        sum_term += self.layers[i_lay + 1].nodes[j].weights[i_node] * delta[0][j]
                    
                    d.append( derivatives[i_lay](a[i_lay][i_node]) * sum_term )
                delta.insert(0, np.array(d))
            #Calculate the gradient of the set of weights and update the weights using gradient descent    
            for i_lay in range(1, len(self.layers)):
                layer = self.layers[i_lay]
                for i_node in range(layer.intercept, len(layer.nodes)):
                    delta_w = h * 2 * delta[i_lay-1][i_node-1] * a[i_lay-1]
                    
                    self.layers[i_lay].nodes[i_node].weights += delta_w

            

if __name__ == "__main__":
    np.random.seed(0)
    X_train, y_train, X_test, y_test = get_data(n_train=280, n_test=120)

    # TODO: Your code goes here.
    network = Feedforward_NN([3,3,1], [1,1,0], [linear_activation, sigmoid_activation, linear_activation])
    
    network.train_NN(X_train,y_train, 0.01 , [d_linear, d_sigmoid, d_linear])
    
    y_hat = np.zeros(len(y_train))
    for i in range(len(X_train)):
        y_hat[i] = network.inference(X_train[i])
    
   
    print(f"The training error is: {MSE(y_hat, y_train)}.")
    print(f"The variance in the training data is {np.std(y_train)**2}.")
    
    y_hat = np.zeros(len(y_test))
    for i in range(len(X_test)):
        y_hat[i] = network.inference(X_test[i])
    
   
    print(f"The test error is: {MSE(y_hat, y_test)}.")
    print(f"The variance in the test data is {np.std(y_test)**2}.") 
    
    
    
