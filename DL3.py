import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.metrics import confusion_matrix


class DLLayer:
    def __init__(self, name, num_units, input_shape, activation="relu", W_initialization="random", learning_rate=1.2,
                 optimization=None):
        # Constant parameters
        self.name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation.lower()
        self._learning_rate = learning_rate
        self._optimization = optimization
        self.alpha = learning_rate
        self.random_scale = 0.01
        self.init_weights(W_initialization)

        # Optimization parameters
        if self._optimization == "adaptive":
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = np.full((self._num_units, *self._input_shape), self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = -0.5

        # Activation
        self.activation_trim = 1e-10

        if self._activation == "sigmoid":
            self.activation_forward = self._sigmoid
            self.activation_backward = self._sigmoid_backward

        elif self._activation == "trim_sigmoid":
            self.activation_forward = self._trim_sigmoid
            self.activation_backward = self._trim_sigmoid_backward

        elif self._activation == "tanh":
            self.activation_forward = self._tanh
            self.activation_backward = self._tanh_backward

        elif self._activation == "trim_tanh":
            self.activation_forward = self._trim_tanh
            self.activation_backward = self._trim_tanh_backward

        elif self._activation == "relu":
            self.activation_forward = self._relu
            self.activation_backward = self._relu_backward

        elif self._activation == "leaky_relu":
            self.leaky_relu_d = 0.01
            self.activation_forward = self._leaky_relu
            self.activation_backward = self._leaky_relu_backward

        elif self._activation == "softmax":
            self.activation_forward = self._soft_max
            self.activation_backward = self._softmax_backward

        elif self._activation == "trim_softmax":
            self.activation_forward = self._trim_softmax
            self.activation_backward = self._softmax_backward

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units, 1), dtype=float)

        if W_initialization.lower() == "zeros":
            self.W = np.full((self._num_units, *self._input_shape), self.alpha)
        elif W_initialization.lower() == "random":
            self.W = np.random.randn(*(self._num_units, *self._input_shape)) * self.random_scale
        elif W_initialization.lower() == "xavier":
            prev_l = self._input_shape[0]
            self.W = np.random.randn(self._num_units, prev_l) * np.sqrt(1 / prev_l)
        elif W_initialization.lower() == "he":
            prev_l = self._input_shape[0]
            self.W = np.random.randn(self._num_units, prev_l) * np.sqrt(2 / prev_l)
        else:
            try:
                with h5py.File(W_initialization, 'r') as hf:
                    self.W = hf['W'][:]
                    self.b = hf['b'][:]
            except FileNotFoundError:
                raise NotImplementedError("Unrecognized initialization:", W_initialization)

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _sigmoid_backward(self, dA):
        A = self._sigmoid(self._Z)
        return dA * A * (1 - A)

    def _trim_sigmoid(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1 / (1 + np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = A = 1 / (1 + np.exp(-Z))
            TRIM = self.activation_trim
            if TRIM > 0:
                A = np.where(A < TRIM, TRIM, A)
                A = np.where(A > 1 - TRIM, 1 - TRIM, A)
            return A

    def _trim_sigmoid_backward(self, dA):
        A = self._trim_sigmoid(self._Z)

        return dA * A * (1 - A)

    def _tanh(self, Z):
        return np.tanh(Z)

    def _tanh_backward(self, dA):
        A = self._tanh(self._Z)
        return dA * (1 - A ** 2)

    def _trim_tanh(self, Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1 + TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)
        return A

    def _trim_tanh_backward(self, dA):
        A = self._trim_tanh(self._Z)

        return dA * (1 - A ** 2)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_backward(self, dA):
        return np.where(self._Z <= 0, 0, dA)

    def _leaky_relu(self, Z):
        return np.where(Z <= 0, Z * self.leaky_relu_d, Z)

    def _leaky_relu_backward(self, dA):
        return np.where(self._Z <= 0, dA * self.leaky_relu_d, dA)

    def _soft_max(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    def _softmax_backward(self, dZ):
        return dZ

    def _trim_softmax(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                eZ = np.exp(Z)
            except FloatingPointError:
                Z = np.where(Z > 100, 100, Z)
                eZ = np.exp(Z)
        A = eZ / np.sum(eZ, axis=0)
        return A

    def forward_propagation(self, A_prev, is_predict):
        self._A_prev = np.array(A_prev, copy=True)
        self._Z = np.dot(self.W, A_prev) + self.b
        A = self.activation_forward(self._Z)

        return A

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = self._A_prev.shape[1]
        self.dW = (1.0 / m) * np.dot(dZ, self._A_prev.T)
        self.db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_Prev = np.dot(self.W.T, dZ)

        return dA_Prev

    def update_parameters(self):
        if self._optimization is None:
            self.W -= self.dW * self.alpha
            self.b -= self.db * self.alpha
        elif self._optimization == "adaptive":
            self._adaptive_alpha_W *= np.where(self._adaptive_alpha_W * self.dW > 0, self.adaptive_cont,
                                               self.adaptive_switch)
            self._adaptive_alpha_b *= np.where(self._adaptive_alpha_b * self.db > 0, self.adaptive_cont,
                                               self.adaptive_switch)
            self.W -= self._adaptive_alpha_W
            self.b -= self._adaptive_alpha_b

    def save_weights(self, path, file_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with h5py.File(f"{path}/{file_name}.h5", 'w') as hf:
            hf.create_dataset('W', data=self.W)
            hf.create_dataset('b', data=self.b)

    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"

        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d) + "\n"

        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"

        # optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont) + "\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch) + "\n"

        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape) + "\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()

        return s


class DLModel:
    def __init__(self, name="Model"):
        self.name = name
        self.layers = [None]
        self._is_compiled = False

    def add(self, layer):
        self.layers.append(layer)

    def _squared_means(self, AL, Y):
        return (AL - Y) ** 2

    def _squared_means_backward(self, AL, Y):
        return 2 * (AL - Y)

    def _cross_entropy(self, AL, Y):
        error = np.where(Y == 0, -np.log(1 - AL), -np.log(AL))
        return error

    def _cross_entropy_backward(self, AL, Y):
        dAL = np.where(Y == 0, 1 / (1 - AL), -1 / AL)
        return dAL

    def _categorical_cross_entropy(self, AL, Y):
        errors = np.where(Y == 1, -np.log(AL), 0)
        return errors

    def _categorical_cross_entropy_backward(self, AL, Y):
        dA = AL - Y
        return dA

    def compile(self, loss, threshold=0.5):
        self.threshold = threshold
        self.loss = loss.lower()

        if "squared" in loss and "means":
            self.loss_forward = self._squared_means
            self.loss_backward = self._squared_means_backward
        elif "categorical" in loss and "cross" in loss and "entropy" in loss:
            self.loss_forward = self._categorical_cross_entropy
            self.loss_backward = self._categorical_cross_entropy_backward
        elif "cross" in loss and "entropy" in loss:
            self.loss_forward = self._cross_entropy
            self.loss_backward = self._cross_entropy_backward

        self._is_compiled = True

    def compute_cost(self, AL, Y):
        m = AL.shape[1]
        errors = self.loss_forward(AL, Y)

        return np.sum(errors) / m

    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []

        for i in range(num_iterations):
            # forward propagation
            Al = X
            for l in range(1, L):
                Al = self.layers[l].forward_propagation(Al, False)

            # backward propagation
            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1, L)):
                dAl = self.layers[l].backward_propagation(dAl)

                # update parameters
                self.layers[l].update_parameters()

            # record progress

            if i > 0 and i % print_ind == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
                print("cost after ", str(i // print_ind), "%:", str(J))
        return costs

    def predict(self, X):
        Al = X
        L = len(self.layers)
        for i in range(1, L):
            Al = self.layers[i].forward_propagation(Al, True)

        if Al.shape[0] > 1:
            return np.where(Al == Al.max(axis=0), 1, 0)
        return Al > self.threshold

    def confusion_matrix(self, X, Y):
        AL = self.predict(X)
        predictions = np.argmax(AL, axis=0)
        labels = np.argmax(Y, axis=0)

        return confusion_matrix(predictions, labels)

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(1, len(self.layers)):
            self.layers[i].save_weights(path, f"Layer{i}")

    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers) - 1) + "\n"

        if self._is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) + "\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1, len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"

        return s
