from Layers import Layer
import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, layers, weight_init = 'uniform', task = 'regression'):
        self.task = task # 'regression' or 'classification'
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
    
    def backpropagate(self, X, y, activations):
        y_pred = activations[-1]

        errors = []
        if self.task == 'classification':
            errors.append(y_pred - y)
        else:
            errors.append((y_pred - y) * self.layers[-1].activation_derivative(activations[-1]))

        for i in range(len(self.layers) - 2, -1, -1):  
            delta = errors[-1].dot(self.layers[i+1].weights.T)
            if self.layers[i].activation != 'softmax':
                delta *= self.layers[i].activation_derivative(activations[i+1])
            errors.append(delta)
        
        errors.reverse()
        
        weight_gradients = []
        bias_gradients = []
        
        for i, layer in enumerate(self.layers):
            dw = activations[i].T @ errors[i] / X.shape[0]
            db = np.mean(errors[i], axis=0, keepdims=True)
            weight_gradients.append(dw)
            bias_gradients.append(db)
        
        return weight_gradients, bias_gradients
    
    def train(self, X, y, epochs, learning_rate, batch_size=None, normalize=True, mode='standard', beta=0.9, epsilon=1e-8):

        velocity = [np.zeros_like(layer.weights) for layer in self.layers]
        cache = [np.zeros_like(layer.weights) for layer in self.layers] 

        if batch_size is None:
            batch_size = len(X)

        if self.task == 'classification':
            y_oh = self.one_hot_encode(y)

        if normalize and self.task == 'regression':
            X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
            y_norm = (y - y.mean(axis=0)) / y.std(axis=0)
        elif normalize and self.task == 'classification':
            X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
            y_norm = y_oh
        else:
            X_norm = X
            y_norm = y_oh if self.task == 'classification' else y
        
        loss_history = []
        weight_history = []

        for epoch in range(epochs):
            
            epoch_predictions = []
            for batch_start in range(0, len(X), batch_size):
                batch_X = X_norm[batch_start:batch_start + batch_size]
                batch_y = y_norm[batch_start:batch_start + batch_size]

                activations = self.feedforward(batch_X)
                epoch_predictions.append(activations[-1])
                weight_gradients, bias_gradients = self.backpropagate(batch_X, batch_y, activations)                

                for j in range(len(self.layers)):
                    gradient_max = 1
                    if mode == 'momentum':
                        velocity[j] = beta * velocity[j] + (1 - beta) * np.clip(weight_gradients[j], -gradient_max, gradient_max)
                        self.layers[j].weights -= learning_rate * velocity[j]
                        self.layers[j].biases -= learning_rate * np.clip(bias_gradients[j], -gradient_max, gradient_max)
                    
                    elif mode == 'rmsprop':
                        cache[j] = beta * cache[j] + (1 - beta) * (np.clip(weight_gradients[j], -gradient_max, gradient_max) ** 2)
                        self.layers[j].weights -= learning_rate * np.clip(weight_gradients[j], -gradient_max, gradient_max) / (np.sqrt(cache[j]) + epsilon)
                        self.layers[j].biases -= learning_rate * np.clip(bias_gradients[j], -gradient_max, gradient_max)

                    else:
                        self.layers[j].weights -= learning_rate * np.clip(weight_gradients[j], -gradient_max, gradient_max)
                        self.layers[j].biases -= learning_rate * np.clip(bias_gradients[j], -gradient_max, gradient_max)


            if (epoch+1) % 100 == 0:
                y_pred = self.predict(X_norm)
                if normalize and self.task == 'regression':
                    y_pred = y_pred * y.std(axis=0) + y.mean(axis=0)
                y_pred_epoch = np.vstack(epoch_predictions)

                loss = self.mse(y, y_pred) if self.task == 'regression' else self.cross_entropy(y_norm, y_pred_epoch)
                loss_history.append(loss)
                #weight_history.append([layer.weights.copy() for layer in self.layers])
                if self.task == 'classification':
                    f1 = self.f1_score(y, y_pred)
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}, F1 Score: {f1}")
                else:  
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
            
        self.plot_loss(loss_history, ((1 * epochs) // 2), epochs)
    
    def predict(self, X):
        if self.task == 'classification':
            return np.argmax(self.feedforward(X)[-1], axis=1).flatten()
        else:
            return self.feedforward(X)[-1]
    
    def set_weights_and_biases(self, layer_idx, weights, biases):
        self.layers[layer_idx].weights = weights
        self.layers[layer_idx].biases = biases

    def one_hot_encode(self, y, n=None):
        y=y.astype(int)
        if n is None:
            n = len(np.unique(y)) 
        one_hot = np.zeros((y.shape[0], n))

        for i in range(y.shape[0]):
            one_hot[i, y[i]] = 1  

        return one_hot
    

    ##### METRICS #####
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def f1_score(self, y_true, y_pred):
        classes = set(y_true) | set(y_pred)
        f1_all = 0

        for cls in classes:
            tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
            fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_true, y_pred))
            fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_true, y_pred))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_all += f1
        
        return f1_all / len(classes) if classes else 0

    
    def cross_entropy(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return loss



    ##### PLOTTING FUNCTIONS #####
    def plot_loss(self, loss_history, start_epoch, end_epoch):
        plt.figure(figsize=(8, 3))

        plt.subplot(1, 2, 1)
        x_values = [i * 100 for i in range(len(loss_history))]
        plt.scatter(x_values, loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        x_values_range = [i * 100 for i in range(start_epoch//100, end_epoch//100)]
        plt.scatter(x_values_range, loss_history[start_epoch//100:end_epoch//100])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs for second half of epochs')
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
        plt.figure(figsize=(8, 3))

        if self.task == 'regression':
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
        else:
            plt.subplot(1, 2, 1)
            plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train)
            plt.title('Train Data')

            if normalize:
                X_train_norm = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
                X_test_norm = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
            else:
                X_train_norm = X_train
                X_test_norm = X_test
            y_train_pred = self.predict(X_train_norm)
            y_test_pred = self.predict(X_test_norm)
            plt.subplot(1, 2, 2)
            plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train_pred)
            plt.title('Train Predictions, F1 Score: {:.3f}'.format(self.f1_score(y_train, y_train_pred)))
            plt.show()

            plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test)
            plt.title('Test Data')

            plt.subplot(1, 2, 2)
            plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test_pred)
            plt.title('Test Predictions, F1 Score: {:.3f}'.format(self.f1_score(y_test, y_test_pred)))
            plt.show()