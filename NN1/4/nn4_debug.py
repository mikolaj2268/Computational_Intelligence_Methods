import os

os.chdir('/Users/mikolajmroz/Developer/Computational_Intelligence_Methods')
print(os.getcwd())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score


def relu(x):
    return np.maximum(0, x)



def relu_derivative(x):
    return np.where(x > 0, 1, 0)



def sigmoid(x):
    x = np.clip(x, -500, 500)  # Avoid overflow
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))




def sigmoid_derivative(sigmoid_output):
    # Assumes that sigmoid_output is the result of sigmoid(x)
    return sigmoid_output * (1 - sigmoid_output)



def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)



def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0)



def cross_entropy(softmax_output, y_true):
    # Assuming y_true is one-hot encoded
    m = y_true.shape[1]  # Number of examples
    log_likelihood = -np.log(softmax_output[y_true.argmax(axis=0), range(m)] + 1e-9)  # Small constant added
    loss = np.sum(log_likelihood) / m
    return loss



def cross_entropy_derivative(softmax_output, y_true):
    corrected_softmax_output = softmax_output - y_true

    return corrected_softmax_output






class MLP:
    def __init__(self, sizes, activation_fn=sigmoid, activation_fn_derivative=sigmoid_derivative):
        self.layer_sizes = sizes
        self.activation_fn = activation_fn
        self.layer_weights = [np.random.randn(y, x) * np.sqrt(2. / x)  for x, y in zip(sizes[:-1], sizes[1:])]
        self.layer_biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.activation_fn_derivative = activation_fn_derivative

    def display_weights_biases(self):
        print("Final Weights and Biases:")
        for layer_index, (weights, biases) in enumerate(zip(self.layer_weights, self.layer_biases)):
            print(f"Layer {layer_index + 1} Weights:\n{weights}")
            print(f"Layer {layer_index + 1} Biases:\n{biases}")

    def propagate_forward(self, input_activation):
        activations = [input_activation]
        for biases, weights in zip(self.layer_biases, self.layer_weights[:-1]):
            input_activation = self.activation_fn(np.dot(weights, input_activation) + biases)
            activations.append(input_activation)
        final_input = np.dot(self.layer_weights[-1], input_activation) + self.layer_biases[-1]
        output_activation = softmax(final_input)
        activations.append(output_activation)
        # change
        return output_activation, activations

    def backward_propagation(self, input_val, true_val):
        weight_gradients = [np.zeros(weight.shape) for weight in self.layer_weights]
        bias_gradients = [np.zeros(bias.shape) for bias in self.layer_biases]

        # Forward pass to get activations
        final_act, activations = self.propagate_forward(input_val)

        # Start with the derivative of the loss function w.r.t. the final activation
        error = cross_entropy_derivative(final_act, true_val)

        # Update gradients for the output layer
        bias_gradients[-1] = error
        weight_gradients[-1] = np.dot(error, activations[-2].T)

        # Backpropagate the error
        for l in range(2, len(self.layer_sizes)):
            # The derivative of the activation function is applied to the output of the activation function
            # from the forward pass, hence 'activations[-l]'
            activation_derivative = self.activation_fn_derivative(activations[-l])

            # Correct error propagation
            error = np.dot(self.layer_weights[-l + 1].T, error) * activation_derivative

            bias_gradients[-l] = error
            weight_gradients[-l] = np.dot(error, activations[-l - 1].T)

        return weight_gradients, bias_gradients

    def update_batch(self, batch, learn_rate, regularization, total_size, optimization_method, beta, epsilon=1e-8):
        gradient_w = [np.zeros(weight.shape) for weight in self.layer_weights]
        gradient_b = [np.zeros(bias.shape) for bias in self.layer_biases]

        for input_val, true_val in batch:
            delta_gradient_w, delta_gradient_b = self.backward_propagation(input_val, true_val)
            gradient_w = [w + dw for w, dw in zip(gradient_w, delta_gradient_w)]
            gradient_b = [b + db for b, db in zip(gradient_b, delta_gradient_b)]

        # Update rule for weights and biases based on the optimization method
        if optimization_method == 'momentum':
            # Momentum initialization
            if not hasattr(self, 'velocity_weights'):
                self.velocity_weights = [np.zeros_like(w) for w in self.layer_weights]
                self.velocity_biases = [np.zeros_like(b) for b in self.layer_biases]

            # Update velocities
            self.velocity_weights = [beta * vw + (1 - beta) * gw / len(batch) for vw, gw in
                                     zip(self.velocity_weights, gradient_w)]
            self.velocity_biases = [beta * vb + (1 - beta) * gb / len(batch) for vb, gb in
                                    zip(self.velocity_biases, gradient_b)]

            # Update weights and biases
            self.layer_weights = [(1 - learn_rate * (regularization / total_size)) * w - learn_rate * vw
                                  for w, vw in zip(self.layer_weights, self.velocity_weights)]
            self.layer_biases = [b - learn_rate * vb for b, vb in zip(self.layer_biases, self.velocity_biases)]
        elif optimization_method == 'rmsprop':
            # RMSprop initialization
            if not hasattr(self, 'squared_gradients_weights'):
                self.squared_gradients_weights = [np.zeros_like(w) for w in self.layer_weights]
                self.squared_gradients_biases = [np.zeros_like(b) for b in self.layer_biases]

            # Update squared gradients
            self.squared_gradients_weights = [beta * sgw + (1 - beta) * (gw ** 2) / len(batch)
                                              for sgw, gw in zip(self.squared_gradients_weights, gradient_w)]
            self.squared_gradients_biases = [beta * sgb + (1 - beta) * (gb ** 2) / len(batch)
                                             for sgb, gb in zip(self.squared_gradients_biases, gradient_b)]

            # Update weights and biases
            self.layer_weights = [(1 - learn_rate * (regularization / total_size)) * w -
                                  learn_rate * gw / (np.sqrt(sgw) + epsilon)
                                  for w, sgw, gw in zip(self.layer_weights, self.squared_gradients_weights, gradient_w)]
            self.layer_biases = [b - learn_rate * gb / (np.sqrt(sgb) + epsilon)
                                 for b, sgb, gb in zip(self.layer_biases, self.squared_gradients_biases, gradient_b)]

    def train(self, training_data, epochs, learn_rate, batch_size, regularization=0.0, optimization_method='rmsprop',
              beta=0.9, epsilon=1e-8, visual_interval=10, X_val=None, y_val=None, target=None, adaptive_learn_rate=True,
              decay_rate=0.1, decay_step=100):
        n = len(training_data)

        # Determine mini-batch size based on whether the batch_size_input is a percentage or fixed value
        if isinstance(batch_size, float):  # If batch_size_input is a float, treat it as a percentage
            batch_size = max(1, min(n, int(n * batch_size / 100)))
        elif isinstance(batch_size, int):  # If batch_size_input is an integer, treat it as a fixed size
            batch_size = max(1, min(n, batch_size))
        else:  # Raise an error if batch_size_input is neither float nor int
            raise ValueError("batch_size_input must be an integer (fixed size) or a float (percentage of dataset)")

        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]

            for mini_batch in mini_batches:
                self.update_batch(mini_batch, learn_rate, regularization, n, optimization_method, beta, epsilon)
            if adaptive_learn_rate:
                # Decay the learning rate every decay_step epochs
                if epoch % decay_step == 0 and epoch > 0:
                    learn_rate *= (1. / (1. + decay_rate * epoch))

            if epoch % visual_interval == 0:
                predictions = np.argmax(np.array([self.propagate_forward(x.reshape(-1, 1))[0] for x in X_val]), axis=1)
                accuracy = np.mean(predictions == y_val)
                print(f'epoch: {epoch}', f'Test accuracy: {accuracy}')
                f1_weighted = f1_score(y_val, predictions, average='weighted')
                print(f"F1 Score (Weighted): {f1_weighted}")

                if f1_weighted > target:
                    break



class DataScaler:
    def __init__(self, method="standardization"):
        self.method = method
        self.min = None
        self.max = None
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        if self.method == "min_max":
            return self.fit_transform_min_max(data)
        elif self.method == "standardization":
            return self.fit_transform_standardization(data)
        else:
            raise ValueError("Unsupported scaling method")

    def transform(self, data):
        if self.method == "min_max":
            return self.transform_min_max(data)
        elif self.method == "standardization":
            return self.transform_standardization(data)
        else:
            raise ValueError("Unsupported scaling method")

    def inverse_transform(self, data):
        if self.method == "min_max":
            return self.inverse_transform_min_max(data)
        elif self.method == "standardization":
            return self.inverse_transform_standardization(data)
        else:
            raise ValueError("Unsupported scaling method")

    def fit_transform_min_max(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return (data - self.min) / (self.max - self.min)

    def transform_min_max(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform_min_max(self, data):
        return data * (self.max - self.min) + self.min

    def fit_transform_standardization(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return (data - self.mean) / self.std

    def transform_standardization(self, data):
        return (data - self.mean) / self.std

    def inverse_transform_standardization(self, data):
        return data * self.std + self.mean



def plot_mse(mse_history):
    plt.plot(mse_history)
    plt.title('MSE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.show()


# Loading data

df_train_rings3_regular = pd.read_csv('./data/classification/rings3-regular-training.csv')
df_test_rings3_regular = pd.read_csv('./data/classification/rings3-regular-test.csv')

df_train_easy = pd.read_csv('./data/classification/easy-training.csv')
df_test_easy = pd.read_csv('./data/classification/easy-test.csv')

df_train_xor3 = pd.read_csv('./data/classification/xor3-training.csv')
df_test_xor3 = pd.read_csv('./data/classification/xor3-test.csv')

### rings 3 regular dataset

scaler_X = DataScaler("standardization")

# Scale features
X1_train_rings = df_train_rings3_regular[['x']].values.reshape(-1, 1)
X1_test_rings = df_test_rings3_regular[['x']].values.reshape(-1, 1)


X2_train_rings = df_train_rings3_regular[['y']].values.reshape(-1, 1)
X2_test_rings = df_test_rings3_regular[['y']].values.reshape(-1, 1)

X_train_rings = np.hstack((X1_train_rings, X2_train_rings))
X_test_rings = np.hstack((X1_test_rings, X2_test_rings))

X_train_rings_scaled = np.hstack((scaler_X.fit_transform(X1_train_rings), scaler_X.fit_transform(X2_train_rings)))
X_test_rings_scaled = np.hstack((scaler_X.transform(X1_test_rings), scaler_X.transform(X2_test_rings)))

y_train_rings = df_train_rings3_regular['c'].values.reshape(-1, 1)
y_test_rings = df_test_rings3_regular['c'].values.reshape(-1, 1)

# Encode the 'c' column into one-hot vectors for the training and test datasets
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded_rings = encoder.fit_transform(y_train_rings)
y_test_encoded_rings = encoder.transform(y_test_rings)

num_classes_rings = y_train_encoded_rings.shape[1]
num_classes_rings

training_data_rings = [
    (X_train_rings[i].reshape(-1, 1), y_train_encoded_rings[i].reshape(-1, 1))
    for i in range(len(X_train_rings))
]

# import warnings
#
# #suppress warnings
# warnings.filterwarnings('ignore')

mlp_rings = MLP(sizes=[2, 10, 10, 3], activation_fn=sigmoid,
                activation_fn_derivative=sigmoid_derivative)  # Example layer setup

# Train the MLP using your training data

mlp_rings.train(training_data=training_data_rings, epochs=1000, learn_rate=0.01, batch_size=64, X_val=X_test_rings,
                y_val=y_test_rings, visual_interval=10, target=0.75, decay_rate=0.01, adaptive_learn_rate=False)