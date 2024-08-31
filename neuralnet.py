import numpy as np
from parameters import Parameters


class NeuralNet:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def __init__(self):
        self.rng = np.random.default_rng()
        self.input_weights = self.rng.normal(0, 1,
                                             size=(Parameters.NUM_OBSERVATIONS, Parameters.NUM_HIDDEN_LAYER_NEURONS))
        self.bias1 = self.rng.normal(0, 1, size=Parameters.NUM_HIDDEN_LAYER_NEURONS)
        self.hidden_weights = self.rng.normal(0, 1, size=(Parameters.NUM_HIDDEN_LAYER_NEURONS, 1))
        self.bias2 = self.rng.normal(0, 1)

    def forward(self, observation):
        assert len(observation) == Parameters.NUM_OBSERVATIONS, ("The observation provided does not match the "
                                                                 "expected observation")
        inputs_to_hidden_layer = np.dot(self.input_weights.T, observation) + self.bias1
        hidden_layer_activations = self.relu(inputs_to_hidden_layer)

        prediction = np.dot(self.hidden_weights.T, hidden_layer_activations) + self.bias2
        return prediction[0], hidden_layer_activations

    def backward(self, state, reward, next_state):
        V_current, hidden_activations_current = self.forward(state)
        V_next, _ = self.forward(next_state)

        error = reward + (Parameters.DISCOUNT_RATE * V_next) - V_current
        delta_hidden_weights = error * hidden_activations_current
        delta_bias2 = error

        delta_hidden_input_weights = self.hidden_weights.T * self.relu_derivative(hidden_activations_current)
        delta_input_weights = np.outer(state, delta_hidden_input_weights * error)
        delta_bias1 = delta_hidden_input_weights * error

        self.hidden_weights += Parameters.LEARNING_RATE * delta_hidden_weights.reshape(-1, 1)
        self.bias2 += Parameters.LEARNING_RATE * delta_bias2
        self.input_weights += Parameters.LEARNING_RATE * delta_input_weights
        self.bias1 += Parameters.LEARNING_RATE * delta_bias1.flatten()

    def add_noise(self, noise_std=0.01):
        """Add Gaussian noise to the weights and biases."""
        self.input_weights += self.rng.normal(0, noise_std, self.input_weights.shape)
        self.bias1 += self.rng.normal(0, noise_std, self.bias1.shape)
        self.hidden_weights += self.rng.normal(0, noise_std, self.hidden_weights.shape)
        self.bias2 += self.rng.normal(0, noise_std)
