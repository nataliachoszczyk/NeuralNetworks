from ActivationFunctions import *
import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation='sigmoid', weight_init='uniform'):
        
        activation_functions = {
            'sigmoid': sigmoid,
            'linear': linear,
            'tanh': tanh
        }
        
        activation_derivative = {
            'sigmoid': sigmoid_derivative,
            'linear': linear_derivative,
            'tanh': tanh_derivative
        }
        
        self.activation = activation_functions.get(activation)
        self.activation_derivative = activation_derivative.get(activation)
        
        if weight_init == 'he':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        elif weight_init == 'xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
        else:
            self.weights = np.random.uniform(0, 1, (input_size, output_size))
        
        self.biases = np.zeros((1, output_size))
    
    def forward(self, X):
        return self.activation(np.dot(X, self.weights) + self.biases)