from copy import deepcopy

import numpy as np


class MultiLayerPerceptron:
    SIGMOID_FUNCTION = 0
    TANH_FUNCTION = 1
    BACKPROPAGATION = 0
    QUICKPROP = 1

    __ACTIVATIONS_FUNCTIONS = {
        SIGMOID_FUNCTION: lambda x: 1 / (1 + np.exp(-x)),
        TANH_FUNCTION: lambda x: np.tanh(x),
    }

    __DERIVATIVES_FOR_ACTIVATION_FUNCTIONS = {
        SIGMOID_FUNCTION: lambda x: x * (1.0 - x),
        TANH_FUNCTION: lambda x: 1 - np.power(np.tanh(x), 2),
    }

    ACTIVATION_FUNCTION = SIGMOID_FUNCTION
    TRAINING_ALGORITHM = BACKPROPAGATION

    def __init__(self, input_vector_size, num_neurons_in_hidden_layers, num_neurons_output, normalization_range=None):
        self.input_vector_size = input_vector_size
        self.normalization_range = normalization_range if normalization_range is not None else [-5, 5]
        self.num_neurons_in_hidden_layers = [x for x in num_neurons_in_hidden_layers if x > 0]
        self.num_neurons_output = num_neurons_output
        self.weights = []
        self.a = []
        self.n = []
        self.s = []
        self.derivatives = []
        self.error_reached_QP = None
        self.last_gradient = None
        self.last_nabla_w = None

    def init_weights(self):
        def get_weights_within_normalization_range(size):
            rng = np.random.default_rng()
            a = self.normalization_range[0]
            b = self.normalization_range[1]
            diff = 0.000001
            return (b - a) * rng.random(size) + (a + diff if a < 0 else a - diff)

        layers = self.num_neurons_in_hidden_layers + [self.num_neurons_output]
        self.weights.append(get_weights_within_normalization_range((layers[0], self.input_vector_size + 1)))
        for i in range(1, len(layers)):
            self.weights.append(get_weights_within_normalization_range((layers[i], layers[i - 1] + 1)))

    def forward_propagate(self, inputs):
        self.a = []
        self.n = []
        activation_values = inputs
        for w in self.weights:
            activation_values = np.insert(activation_values, 0, -1)
            activation_values = activation_values.reshape((activation_values.shape[0], 1))
            self.a.append(activation_values)
            net_inputs = np.dot(w, activation_values)
            self.n.append(net_inputs)
            activation_values = self._activation_function(net_inputs)
        self.a.append(activation_values)
        return activation_values

    def back_propagate(self, error):
        self.s = []
        self.derivatives = []
        num_layers = len(self.a)
        output_layer = num_layers - 1
        for i in reversed(range(1, num_layers)):
            is_output_layer = i == output_layer
            a = self.a[i] if is_output_layer else np.delete(self.a[i], 0, 0)
            d = self._activation_function_derivative(a)
            derivatives = np.diag(d.reshape((d.shape[0],)))
            self.derivatives = [derivatives] + self.derivatives
            if is_output_layer:
                s = np.dot(derivatives, error)
                self.s.append(s)
            else:
                weights = np.delete(self.weights[i], 0, 1)
                jacobian_matrix = np.dot(derivatives, weights.T)
                s = np.dot(jacobian_matrix, self.s[0])
                self.s = [s] + self.s

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            new_w = self.weights[i] + learning_rate * np.dot(self.s[i], self.a[i].T)
            self.weights[i] = new_w

    def quickprop(self, learning_rate):
        miu = 1.75  # Recommended value by Scott Fahlman
        is_first_QP_iteration = self.last_gradient is None
        if is_first_QP_iteration:
            self.last_gradient = []
            self.last_nabla_w = []
        for i in range(len(self.weights)):
            gradient = np.dot(self.s[i], self.a[i].T)
            if is_first_QP_iteration:  # Standard BackPropagation
                nabla_w = learning_rate * gradient
                self.last_gradient.append(deepcopy(gradient))
                self.last_nabla_w.append(deepcopy(nabla_w))
            else:
                divisor = self.last_gradient[i] - gradient
                divisor[divisor == 0] = 0.0001
                delta = gradient / divisor
                temp = delta * self.last_nabla_w[i]
                maximum_growth_factor = miu * self.last_nabla_w[i]
                # if temp > miu * last gradient:
                temp_is_greater_than_miu = temp > maximum_growth_factor
                temp[temp_is_greater_than_miu] = maximum_growth_factor[temp_is_greater_than_miu]
                last_gradient_and_current_gradient_product = self.last_gradient[i] * gradient
                gradient_descent = learning_rate * gradient
                nabla_w = temp + gradient_descent
                # if last gradient * current gradient < 0:
                gradients_product_is_less_than_zero = last_gradient_and_current_gradient_product < np.zeros(
                    last_gradient_and_current_gradient_product.shape
                )
                nabla_w[gradients_product_is_less_than_zero] = temp[gradients_product_is_less_than_zero]
                nabla_too_small = nabla_w < np.ones(nabla_w.shape) * 1e-9
                nabla_w[nabla_too_small] = gradient_descent[nabla_too_small]
                self.last_gradient[i] = deepcopy(gradient)
                self.last_nabla_w[i] = deepcopy(nabla_w)
            new_w = self.weights[i] + nabla_w
            self.weights[i] = new_w

    def fit(self, inputs, desired_outputs, epochs, learning_rate, desired_error=None, plotter=None):
        converged = False
        cumulative_error = desired_error if desired_error else 1
        starting_epoch = plotter.current_epoch + 1
        last_epoch = starting_epoch + epochs
        error_reached = [0, 0]
        for epoch in range(starting_epoch, last_epoch):
            if plotter:
                plotter.current_epoch = epoch
            if desired_error and cumulative_error < desired_error:
                converged = True
                break
            cumulative_error = 0
            for _input, desired_output in zip(inputs, desired_outputs):
                output = self.forward_propagate(_input)
                desired_output = desired_output.reshape(output.shape)
                error = desired_output - output
                squared_error = np.dot(error.T, error)
                self.back_propagate(error)
                cumulative_error += squared_error[0][0]
                if self.TRAINING_ALGORITHM == MultiLayerPerceptron.BACKPROPAGATION:
                    self.gradient_descent(learning_rate)
                else:
                    self.quickprop(learning_rate)
                    error_reached[0] = cumulative_error
                    error_reached[1] = epoch
            if plotter:
                plotter.plot_errors(cumulative_error)
            print(f"[{'QP' if self.TRAINING_ALGORITHM == MultiLayerPerceptron.QUICKPROP else 'STD'}] "
                  f"Error at epoch {epoch}: {cumulative_error}")
        if self.error_reached_QP and self.TRAINING_ALGORITHM == MultiLayerPerceptron.BACKPROPAGATION:
            print(f"Error reached with QP: {self.error_reached_QP[0]} within {self.error_reached_QP[1]} epochs")
        else:
            self.error_reached_QP = error_reached
        return converged

    def guess(self, _input, discrete_output=True):
        output = self.forward_propagate(_input)
        if discrete_output:
            output = np.array([1 if x >= 0.5 else 0 for x in output])
        else:
            output = output.flatten()
        return output

    def _activation_function(self, x):
        return self.__ACTIVATIONS_FUNCTIONS[self.ACTIVATION_FUNCTION](x)

    def _activation_function_derivative(self, x):
        return self.__DERIVATIVES_FOR_ACTIVATION_FUNCTIONS[self.ACTIVATION_FUNCTION](x)
