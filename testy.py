import pandas as pd
import numpy as np

df_test = pd.read_csv("./data/regression/square-simple-test.csv")

x_test = np.array(df_test['x'])
y_test = np.array(df_test['y'])

y_test2 = np.array([y + 130 for y in y_test])

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation= tf.nn.sigmoid, input_shape=(1,)),
    #tf.keras.layers.Dense(5, activation= tf.nn.sigmoid, input_shape=(1,)),
    tf.keras.layers.Dense(1, activation= 'linear')
])
optimizer = tf.compat.v1.train.GradientDescentOptimizer(
    1e-2, use_locking=False, name='GradientDescent'
)

model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(x_test, y_test2, epochs=5000, verbose= 2)
print(model.evaluate(x_test, y_test2))
print(model.get_weights())


def weights_generator(layers_n, neural_n, min, max):
    weights_dict = {}
    bias_dict = {}

    layer_0_weights = np.asmatrix(np.random.randint(-10, 12, size=(neural_n, 1)))
    layer_0_bias = np.random.randint(-20, 0, size=(1, neural_n))

    weights_dict[0] = layer_0_weights
    bias_dict[0] = layer_0_bias

    for i in range(layers_n):
        layer_weights = np.asmatrix(np.random.uniform(min, max, size=(neural_n, neural_n)))
        layer_bias = np.random.uniform(0, 0.01, size=(1, neural_n))
        weights_dict[1 + i] = layer_weights
        bias_dict[1 + i] = layer_bias

    layer_n_weights = np.asmatrix(np.random.randint(90, 140, size=(1, neural_n)))
    layer_n_bias = np.array([-130])

    weights_dict[layers_n] = layer_n_weights
    bias_dict[layers_n] = layer_n_bias

    return weights_dict, bias_dict


def broot_force_generator(x_test, layers_n, neural_n):
    while True:

        weights_dict, bias_dict = weights_generator(layers_n, neural_n, -2, 2)
        y_pred = []
        for x in x_test:
            input_n, output_n, hidden_n, hidden_dict = generate_arg(np.array([x]), weights_dict, bias_dict)
            neural_network = NeuralNetwork(
                input_vector=np.array([x]),
                input_n=input_n,
                hidden_n=hidden_n,
                output_n=output_n,
                hidden_dict=hidden_dict,
                weights_dict=weights_dict,
                bias_dict=bias_dict,
                activation_function=sig
            )
            y_pred.append(float(neural_network.feedforward()))

        print(mse(y_test, y_pred))
        if mse(y_test, y_pred) < 1000:
            print(weights_dict)
            print('-----------BIAS---------------')
            print(bias_dict)
            return weights_dict, bias_dict