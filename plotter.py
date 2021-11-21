import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import ListedColormap
from matplotlib.widgets import TextBox, Button

from constants import *
from multilayerperceptron import MultiLayerPerceptron


def new_text_box(axes, prompt):
    return TextBox(plt.axes(axes), prompt)


def new_button(axes, prompt):
    return Button(plt.axes(axes), prompt)


class Plotter:
    X, Y = np.array([]), []
    mlp = None
    learning_rate = 0
    max_epochs = 0
    current_epoch = 0
    current_epoch_text = None
    algorithm_convergence_text = None
    mlp_weights_initialized = False
    quickprop_done = False
    mlp_fitted = False
    mlp_errors = None
    done = False
    mlp_decision_boundaries = []
    mlp_sweep = None
    current_data_set = 0

    def __init__(self):
        self.fig, (self.ax_main, self.ax_errors) = plt.subplots(SUBPLOT_ROWS, SUBPLOT_COLS)
        self.fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT, forward=True)
        plt.subplots_adjust(bottom=SUBPLOT_BOTTOM_ADJUST)
        self.ax_main.set_xlim(NORMALIZATION_RANGE)
        self.ax_main.set_ylim(NORMALIZATION_RANGE)
        self.fig.suptitle(FIG_SUPERIOR_TITLE % QP_ALGORITHM_NAME)
        self.ax_main.set_title(MAIN_SUBPLOT_TITLE)
        self.ax_errors.set_title(ERRORS_SUBPLOT_TITLE)
        self.ax_errors.set_xlabel(ERRORS_SUBPLOT_XLABEL)
        self.ax_errors.set_ylabel(ERRORS_SUBPLOT_YLABEL)

        self.text_box_neurons_hidden_layers = new_text_box(
            TEXT_BOX_NEURONS_HIDDEN_LAYERS_AXES,
            TEXT_BOX_NEURONS_HIDDEN_LAYERS_PROMPT
        )
        self.text_box_learning_rate = new_text_box(TEXT_BOX_LEARNING_RATE_AXES, TEXT_BOX_LEARNING_RATE_PROMPT)
        self.text_box_max_epochs = new_text_box(TEXT_BOX_MAX_EPOCHS_AXES, TEXT_BOX_MAX_EPOCHS_PROMPT)
        self.text_box_desired_error = new_text_box(TEXT_BOX_DESIRED_ERROR_AXES, TEXT_BOX_DESIRED_ERROR_PROMPT)
        button_weights = new_button(BUTTON_WEIGHTS_AXES, BUTTON_WEIGHTS_TEXT)
        button_mlp = new_button(BUTTON_MLP_AXES, BUTTON_MLP_TEXT)
        self.button_data_set = new_button(
            plt.axes(BUTTON_SWITCH_DATA_SET_AXES),
            BUTTON_SWITCH_DATA_SET_TEXT % (self.current_data_set + 1)
        )
        self.text_box_neurons_hidden_layers.on_submit(self.__submit_neurons_hidden_layers)
        self.text_box_max_epochs.on_submit(self.__submit_max_epochs)
        self.text_box_learning_rate.on_submit(self.__submit_learning_rate)
        self.text_box_desired_error.on_submit(self.__submit_desired_error)
        button_weights.on_clicked(self.__initialize_weights)
        button_mlp.on_clicked(self.__fit_mlp)
        self.button_data_set.on_clicked(self.__switch_data_set)
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        plt.show()

    def __initialize_weights(self, event):
        mlp_architecture_defined = self.neurons_hidden_layers is not None
        points_plotted = len(self.X) > 0
        if mlp_architecture_defined and points_plotted:
            self.mlp = MultiLayerPerceptron(
                2,
                self.neurons_hidden_layers,
                NEURONS_OUTPUT_LAYER,
                NORMALIZATION_RANGE
            )
            self.mlp.init_weights()
            self.plot_decision_boundaries(self.mlp.weights[0])
            self.mlp_weights_initialized = True
            self.mlp.TRAINING_ALGORITHM = MultiLayerPerceptron.QUICKPROP

    def __fit_mlp(self, event):
        for boundary in self.mlp_decision_boundaries:
            boundary.remove()
        self.mlp_decision_boundaries = []
        learning_rate_initialized = self.learning_rate != 0
        max_epochs_initialized = self.max_epochs != 0
        desired_error_is_set = self.desired_error != 0.0
        hyper_params_are_set = learning_rate_initialized == max_epochs_initialized == desired_error_is_set is True
        if not self.mlp_fitted and self.mlp_weights_initialized and hyper_params_are_set:
            self.Y = np.array(self.Y)
            converged = self.mlp.fit(self.X, self.Y, self.max_epochs, self.learning_rate, self.desired_error, self)
            convergence_text = ALGORITHM_CONVERGED_TEXT if converged else ALGORITHM_DIDNT_CONVERGE_TEXT
            if self.algorithm_convergence_text:
                self.algorithm_convergence_text.set_text(convergence_text)
            else:
                self.algorithm_convergence_text = self.ax_main.text(
                    CONVERGENCE_TEXT_X_POS,
                    CONVERGENCE_TEXT_Y_POS,
                    convergence_text,
                    fontsize=CONVERGENCE_TEXT_FONT_SIZE
                )
            self.current_epoch_text.set_text(CURRENT_EPOCH_TEXT % self.current_epoch)
            plt.pause(MAIN_SUBPLOT_PAUSE_INTERVAL)
            if self.quickprop_done:
                self.mlp_fitted = True
            else:
                self.mlp.TRAINING_ALGORITHM = MultiLayerPerceptron.BACKPROPAGATION
                self.quickprop_done = True
            self.algorithm_convergence_text.set_text(None)
        self.plot_decision_regions()
        plt.pause(MAIN_SUBPLOT_PAUSE_INTERVAL)
        if self.quickprop_done and not self.mlp_fitted:
            for item in self.mlp_sweep.collections:
                item.remove()
            self.mlp_sweep = None
            self.fig.suptitle(FIG_SUPERIOR_TITLE % STD_ALGORITHM_NAME)

    def __switch_data_set(self, event):
        self.current_data_set = not self.current_data_set
        self.button_data_set.label.set_text(BUTTON_SWITCH_DATA_SET_TEXT % (self.current_data_set + 1))

    def plot_decision_regions(self, points=250):
        cmap = ListedColormap(BASE_COLORS[:2 ** self.mlp.num_neurons_output])

        coords = np.linspace(NORMALIZATION_RANGE[0], NORMALIZATION_RANGE[1], points)
        x, y = np.meshgrid(coords, coords)

        Z = np.array([self.mlp.guess(x) for x in np.array([x.ravel(), y.ravel()]).T])
        new_z = np.array([])
        for z in Z:
            guess_index_string = ""
            for n in z:
                guess_index_string += str(n)
            guess_index = int(guess_index_string, 2)
            new_z = np.append(new_z, guess_index)
        Z = new_z.reshape(x.shape)
        self.mlp_sweep = self.ax_main.contourf(x, y, Z, alpha=0.2, cmap=cmap)

    def plot_decision_boundaries(self, weights):
        creating_boundaries = len(self.mlp_decision_boundaries) == 0
        if creating_boundaries:
            self.current_epoch_text = self.ax_main.text(CURRENT_EPOCH_TEXT_X_POS, CURRENT_EPOCH_TEXT_Y_POS,
                                                        CURRENT_EPOCH_TEXT % self.current_epoch,
                                                        fontsize=CURRENT_EPOCH_TEXT_FONT_SIZE)
        else:
            self.current_epoch_text.set_text(CURRENT_EPOCH_TEXT % self.current_epoch)
        x1 = np.array([self.X[:, 0].min() - 2, self.X[:, 0].max() + 2])
        for i in range(len(weights)):
            weight = weights[i]
            m = -weight[1] / weight[2]
            c = weight[0] / weight[2]
            x2 = m * x1 + c
            # Plotting
            if creating_boundaries:
                decision_boundary, = self.ax_main.plot(
                    x1,
                    x2,
                    DECISION_BOUNDARIES_MARKERS[i % len(DECISION_BOUNDARIES_MARKERS)]
                )
                self.mlp_decision_boundaries.append(decision_boundary)
            else:
                self.mlp_decision_boundaries[i].set_xdata(x1)
                self.mlp_decision_boundaries[i].set_ydata(x2)
            self.fig.canvas.draw()
            plt.pause(MAIN_SUBPLOT_PAUSE_INTERVAL)

    def plot_errors(self, cumulative_error):
        is_qp = int(not self.quickprop_done)
        if not self.mlp_errors:
            self.mlp_errors = [[[], []], [[], []]]
        else:
            self.ax_errors.clear()
        self.ax_errors.set_title(ERRORS_SUBPLOT_TITLE)
        self.ax_errors.set_xlabel(ERRORS_SUBPLOT_XLABEL)
        self.ax_errors.set_ylabel(ERRORS_SUBPLOT_YLABEL)
        self.mlp_errors[is_qp][0].append(self.current_epoch)
        self.mlp_errors[is_qp][1].append(cumulative_error)
        if self.quickprop_done:
            self.ax_errors.plot(
                self.mlp_errors[1][0],
                self.mlp_errors[1][1],
                QP_ERRORS_PLOT_MARKER
            )
        self.ax_errors.plot(
            self.mlp_errors[is_qp][0],
            self.mlp_errors[is_qp][1],
            STD_ERRORS_PLOT_MARKER if self.quickprop_done else QP_ERRORS_PLOT_MARKER
        )
        plt.pause(ERRORS_SUBPLOT_PAUSE_INTERVAL)

    def __onclick(self, event):
        if event.inaxes == self.ax_main:
            current_point = [event.xdata, event.ydata]
            is_right_click = event.button == MouseButton.RIGHT
            marker_index_string = str(int(self.current_data_set)) + str(int(is_right_click))
            marker_index = int(marker_index_string, 2)
            if self.mlp_fitted:
                guess = self.mlp.guess(current_point)
                guess_index_string = ""
                for x in guess:
                    guess_index_string += str(x)
                guess_index = int(guess_index_string, 2)
                self.ax_main.plot(event.xdata, event.ydata, CLASSES_MARKERS_POST_FIT[guess_index])
            else:
                self.X = np.append(self.X, current_point).reshape([len(self.X) + 1, 2])
                # Data set 0:
                # Left click = Class 0 - Right click = Class 1
                # Data set 1:
                # Left click = Class 2 - Right click = Class 3
                self.Y.append([int(x) for x in marker_index_string])
                self.ax_main.plot(event.xdata, event.ydata, CLASSES_MARKERS_PRE_FIT[marker_index])
            self.fig.canvas.draw()

    def __check_if_valid_expression(self, expression, text_box, default_value):
        value = 0
        try:
            value = eval(expression)
        except (SyntaxError, NameError):
            if expression:
                value = default_value
                text_box.set_val(value)
        finally:
            return value

    def __submit_neurons_hidden_layers(self, expression):
        result = self.__check_if_valid_expression(
            expression,
            self.text_box_neurons_hidden_layers,
            NEURONS_HIDDEN_LAYERS
        )
        if type(result) == tuple:
            self.neurons_hidden_layers = [x for x in result]
        else:
            self.neurons_hidden_layers = [result]

    def __submit_learning_rate(self, expression):
        self.learning_rate = self.__check_if_valid_expression(expression, self.text_box_learning_rate, LEARNING_RATE)

    def __submit_max_epochs(self, expression):
        self.max_epochs = self.__check_if_valid_expression(expression, self.text_box_max_epochs, MAX_EPOCHS)

    def __submit_desired_error(self, expression):
        self.desired_error = self.__check_if_valid_expression(expression, self.text_box_desired_error, DESIRED_ERROR)


if __name__ == '__main__':
    Plotter()
