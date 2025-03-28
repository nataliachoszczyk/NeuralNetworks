import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def linear(z):
    return z

def linear_derivative(z):
    return np.ones_like(z)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - z**2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)
