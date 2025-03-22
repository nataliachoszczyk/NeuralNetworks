import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def linear(z):
    return z

class MLP:
    def __init__(self, layer_sizes,
                 hidden_activation = 'sigmoid',
                 output_activation = 'linear'):
        
        self.layer_sizes = layer_sizes
        
        activation_functions = {
            'sigmoid': sigmoid,
            'linear': linear
        }
        
        self.hidden_activation = activation_functions.get(hidden_activation)
        self.output_activation = activation_functions.get(output_activation)
        
        # domyślne wartości wag i biasów losowe
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
        
    def forward(self, X):
        """
        przejście w przód przez sieć
        X: macierz wejść, każdy wiersz to jedna obserwacja
        return: lista kolejnych aktywacji i ostateczne wyjście
        """
        activations = [X]
        a = X
        
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.hidden_activation(z)
            activations.append(a)
            
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        a = self.output_activation(z)
        activations.append(a)
        
        return activations
    
    def predict(self, X):
        return self.forward(X)[-1]
    
    def set_weights_and_biases(self, layer_idx, weights, biases):
        self.weights[layer_idx] = weights
        self.biases[layer_idx] = biases
        
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
        