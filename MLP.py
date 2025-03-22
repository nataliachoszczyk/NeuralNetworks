from Layers import Layer
import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layers, weight_init = 'uniform'):
        self.layers = []
        for layer in layers:
            self.layers.append(Layer(layer['input_size'], layer['output_size'], layer['activation'], weight_init))
    
    def feedforward(self, X):
        activations = [X]
        a = X
        for layer in self.layers:
            a = layer.forward(a)
            activations.append(a)
        return activations
    
    def backpropagate(self, X, y):
        activations = self.feedforward(X)
        y_pred = activations[-1]
        errors = [y_pred - y]
        
        for i in range(len(self.layers) - 1, 0, -1):
            errors.append(errors[-1].dot(self.layers[i].weights.T) * self.layers[i-1].activation_derivative(activations[i]))
        errors.reverse()
        
        weight_gradients = []
        bias_gradients = []
        
        for i, layer in enumerate(self.layers):
            dw = activations[i].T @ errors[i] / X.shape[0]
            db = np.mean(errors[i], axis=0, keepdims=True)
            weight_gradients.append(dw)
            bias_gradients.append(db)
        
        return weight_gradients, bias_gradients
    
    def train(self, X, y, epochs, learning_rate, batch_size=None, normalize=True):
        if batch_size is None:
            batch_size = len(X)
        
        if normalize:
        # normalizacja danych
            X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
            y_norm = (y - y.mean(axis=0)) / y.std(axis=0)
        else:
            X_norm = X
            y_norm = y
        
        loss_history = []
        weight_history = []
        for epoch in range(epochs):
            
            for batch_start in range(0, len(X), batch_size):
                batch_X = X_norm[batch_start:batch_start + batch_size]
                batch_y = y_norm[batch_start:batch_start + batch_size]

                # weight_gradients_sum = [np.zeros_like(layer.weights) for layer in self.layers]
                # bias_gradients_sum = [np.zeros_like(layer.biases) for layer in self.layers]

                # # Przechodzimy przez próbki w batchu
                # for i in range(len(batch_X)):
                #     weight_gradients, bias_gradients = self.backpropagate(batch_X[i:i+1], batch_y[i:i+1])

                #     for j in range(len(self.layers)):
                #         weight_gradients_sum[j] += weight_gradients[j]
                #         bias_gradients_sum[j] += bias_gradients[j]
                
                weight_gradients, bias_gradients = self.backpropagate(batch_X, batch_y)                

                # Aktualizacja wag i biasów po każdej porcji batch_size
                for j in range(len(self.layers)):
                    gradient_max = 1
                    self.layers[j].weights -= learning_rate * np.clip(weight_gradients[j], -gradient_max, gradient_max)
                    self.layers[j].biases -= learning_rate * np.clip(bias_gradients[j], -gradient_max, gradient_max)
                    # self.layers[j].weights -= learning_rate * weight_gradients[j]
                    # self.layers[j].biases -= learning_rate * bias_gradients[j]

            y_pred = self.predict(X_norm)
            if normalize:
                y_pred = y_pred * y.std(axis=0) + y.mean(axis=0)
            loss = self.mse(y, y_pred)
            loss_history.append(loss)
            weight_history.append([layer.weights.copy() for layer in self.layers])
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
            
        self.plot_loss(loss_history)
        self.plot_weights(weight_history)
    
    def predict(self, X):
        return self.feedforward(X)[-1]
    
    def set_weights_and_biases(self, layer_idx, weights, biases):
        self.layers[layer_idx].weights = weights
        self.layers[layer_idx].biases = biases
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def plot_loss(self, loss_history):
        plt.scatter(range(len(loss_history)), loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.grid(True)
        plt.show()
    
    def plot_weights(self, weight_history):
        for layer_idx in range(len(weight_history[0])):
            plt.figure(figsize=(6, 4))
            weights = np.array([weight_history[epoch][layer_idx] for epoch in range(len(weight_history))])

            for i in range(weights.shape[1]):
                plt.plot(range(len(weight_history)), weights[:, i], label=f'Weight {i + 1}')
            
            plt.xlabel('Epoch')
            plt.ylabel('Weight Value')
            plt.title(f'Layer {layer_idx + 1} Weights vs Epochs')
            plt.grid(True)
            plt.show()
        
    def plot_predictions(self, X_train, y_train, X_test, y_test, normalize=True):
        plt.figure(figsize=(12, 5))

        # Wykres dla zbioru treningowego
        plt.subplot(1, 2, 1)
        if normalize:
            X_train_norm = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        else:
            X_train_norm = X_train
        y_pred_train_norm = self.predict(X_train_norm)
        if normalize:
            y_pred_train = y_pred_train_norm * y_train.std() + y_train.mean()
        else:
            y_pred_train = y_pred_train_norm
        train_mse = self.mse(y_train, y_pred_train)
        plt.scatter(X_train, y_train, label="Train Data")
        plt.scatter(X_train, y_pred_train, label="Predicted")
        plt.title(f'Training Data vs Predictions (MSE: {train_mse:.4f})')
        plt.legend()
        plt.grid(True)

        # Wykres dla zbioru testowego
        plt.subplot(1, 2, 2)
        if normalize:
            X_test_norm = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
        else:
            X_test_norm = X_test
        y_pred_test_norm = self.predict(X_test_norm)
        if normalize:
            y_pred_test = y_pred_test_norm * y_train.std() + y_train.mean()
        else:
            y_pred_test = y_pred_test_norm
        test_mse = self.mse(y_test, y_pred_test)
        plt.scatter(X_test, y_test, label="Test Data")
        plt.scatter(X_test, y_pred_test, label="Predicted")
        plt.title(f'Test Data vs Predictions (MSE: {test_mse:.4f})')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()