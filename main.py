import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


np.random.seed(42)  # Example of setting a seed for NumPy


class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) * np.sqrt(2. / x)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        # self.weights = [np.array([[0.70245989],
        #                           [-0.19553525],
        #                           [0.91596991],
        #                           [2.15388948],
        #                           [-0.33114288]]),
        #                 np.array([[-0.14808121, 0.99878188, 0.48536834, -0.29692167, 0.3431451],
        #                           [-0.29309108, -0.29455336, 0.15303038, -1.21006468, -1.09093383],
        #                           [-0.35562186, -0.64057065, 0.19874746, -0.57428485, -0.89321929],
        #                           [0.92695767, -0.14279347, 0.04270859, -0.90108987, -0.34429787],
        #                           [0.07015361, -0.72795226, 0.23761229, -0.37987726, -0.18448333]]),
        #                 np.array([[-0.38055268, 1.17148358, -0.00853639, -0.66895513, 0.52022308]])]
        print(self.weights)
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    def print_final_weights_and_biases(self):
        print("Final Weights and Biases:")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            print(f"Layer {i + 1} Weights:\n{w}")
            print(f"Layer {i + 1} Biases:\n{b}")

    def feedforward(self, a):
        activations = [a]  # Stores all activations
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)
            activations.append(a)
        # Linear activation for the last layer
        a = np.dot(self.weights[-1], a) + self.biases[-1]
        activations.append(a)
        return activations[-1], activations  # Return final activation and all activations

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        final_output, activations = self.feedforward(x)
        zs = [np.dot(w, act) + b for w, b, act in zip(self.weights, self.biases, activations[:-1])]  # Z values

        # Output layer error
        delta = self.cost_derivative(final_output, y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate the error
        for l in range(2, len(self.layer_sizes)):
            sp = sigmoid_derivative(zs[-l])
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_w, nabla_b

    def update_mini_batch(self, mini_batch, learning_rate, lambda_, n):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        # Update weights with L2 regularization
        self.weights = [(1 - learning_rate * (lambda_ / n)) * w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, training_data, epochs, learning_rate, batch_size, lambda_=0.0, update_method='batch',
              plot_interval=None):
        n = len(training_data)
        learning_rate_init = learning_rate
        for j in range(epochs):
            # Plot weights at the specified interval
            if plot_interval and j % plot_interval == 0:
                print(f"Epoch {j}:")
                self.plot_weights()

            np.random.shuffle(training_data)
            if update_method == 'batch':
                mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, learning_rate, lambda_, n)
            elif update_method == 'epoch':
                self.update_mini_batch(training_data, learning_rate, lambda_, n)
            # Learning rate schedule
            learning_rate = learning_rate_init / (1 + 0.01 * j)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


df_train_square = pd.read_csv('./data/regression/square-simple-training.csv')
X_train_square = df_train_square['x'].values.reshape(-1, 1)
y_train_square = df_train_square['y'].values.reshape(-1, 1)
# Trenowanie sieci
mlp_square_1_5 = MLP([1, 5, 5, 1])
training_data = [(x.reshape(-1, 1), y) for x, y in zip(X_train_square, y_train_square)]
training_data
mlp_square_1_5.train(training_data, epochs=1000, learning_rate=0.01, batch_size=10)

# df_test_square = pd.read_csv('mio1/regression/square-simple-test.csv')
X_test_square = df_test_square['x'].values.reshape(-1, 1)
y_test_square = df_test_square['y'].values.reshape(-1, 1)

# Generate predictions
predictions = np.array([mlp_square_1_5.feedforward(x.reshape(-1, 1))[0] for x in X_test_square])

# Flatten predictions to ensure it has the same shape as y_test
predictions = predictions.reshape(-1, 1)

# Calculate MSE score
for i in range(len(predictions)):
    print(predictions[i], y_test_square[i])
mse_score = mse(predictions, y_test_square)

print(f"MSE Score: {mse_score}")

mlp_square_1_5.print_final_weights_and_biases()
