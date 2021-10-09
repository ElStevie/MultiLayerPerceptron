NORMALIZATION_RANGE = -5.0, 5.0
LEARNING_RATE = 0.2
MAX_EPOCHS = 30
DESIRED_ERROR = 0.1
NEURONS_INPUT_LAYER = 2
NEURONS_HIDDEN_LAYERS = 0
NEURONS_OUTPUT_LAYER = 2

FIG_WIDTH = 13
FIG_HEIGHT = 7
SUBPLOT_ROWS = 1
SUBPLOT_COLS = 2
SUBPLOT_BOTTOM_ADJUST = 0.35
FIG_SUPERIOR_TITLE = "Backpropagation training algorithm"
MAIN_SUBPLOT_TITLE = "Multilayer Perceptron"
ERRORS_SUBPLOT_TITLE = "Cumulative error"
ERRORS_SUBPLOT_XLABEL = "Epochs"
ERRORS_SUBPLOT_YLABEL = "E²"

MAIN_SUBPLOT_PAUSE_INTERVAL = 0.1
ERRORS_SUBPLOT_PAUSE_INTERVAL = 0.3

ALGORITHM_CONVERGED_TEXT = 'Algorithm converged'
ALGORITHM_DIDNT_CONVERGE_TEXT = "Algorithm couldn't converge"
CONVERGENCE_TEXT_FONT_SIZE = 10
CONVERGENCE_TEXT_X_POS = -0.25
CONVERGENCE_TEXT_Y_POS = 0.9
CONVERGENCE_OFFSET = 0.25

CURRENT_EPOCH_TEXT = 'Epoch: %s'
CURRENT_EPOCH_TEXT_FONT_SIZE = 10
CURRENT_EPOCH_TEXT_X_POS = NORMALIZATION_RANGE[1] * 0.6
CURRENT_EPOCH_TEXT_Y_POS = NORMALIZATION_RANGE[1] * 0.9

BASE_COLORS = ['b', 'r', 'g', 'm', 'c', 'y']
CLASSES_MARKERS = ['x', '.', '^', 's']
CLASSES_MARKERS_PRE_FIT = [color + marker for marker, color in zip(CLASSES_MARKERS, BASE_COLORS)]
CLASSES_MARKERS_POST_FIT = ['k' + marker for marker in CLASSES_MARKERS]

DECISION_BOUNDARIES_MARKERS = [c + '-' for c in reversed(BASE_COLORS)]
ERRORS_PLOT_MARKER = 'm-'

TEXT_BOX_NEURONS_INPUT_LAYER_AXES = [0.25, 0.2, 0.1, 0.05]
TEXT_BOX_NEURONS_INPUT_LAYER_PROMPT = "Neurons in input layer:"
TEXT_BOX_NEURONS_HIDDEN_LAYERS_AXES = [0.5, 0.2, 0.175, 0.05]
TEXT_BOX_NEURONS_HIDDEN_LAYERS_PROMPT = "Neurons in hidden layers:"
BUTTON_WEIGHTS_AXES = [0.75, 0.2, 0.125, 0.05]
BUTTON_WEIGHTS_TEXT = "Initialize Weights"
TEXT_BOX_LEARNING_RATE_AXES = [0.225, 0.125, 0.15, 0.05]
TEXT_BOX_LEARNING_RATE_PROMPT = "Learning rate (η):"
TEXT_BOX_MAX_EPOCHS_AXES = [0.5, 0.125, 0.15, 0.05]
TEXT_BOX_MAX_EPOCHS_PROMPT = "Max no. of epochs:"
TEXT_BOX_DESIRED_ERROR_AXES = [0.75, 0.125, 0.15, 0.05]
TEXT_BOX_DESIRED_ERROR_PROMPT = "Desired error:"
BUTTON_SWITCH_DATA_SET_AXES = [0.25, 0.05, 0.15, 0.05]
BUTTON_SWITCH_DATA_SET_TEXT = "Current data set: %d"
BUTTON_MLP_AXES = [0.5, 0.05, 0.1, 0.05]
BUTTON_MLP_TEXT = "MLP"
