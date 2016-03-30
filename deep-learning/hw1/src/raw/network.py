from src.cost_funcs import *

from src.functions.activation_funcs import *
from src.functions.activation_funcs import dummy_a


class Network(object):
    def __init__(self, sizes, act_funcs, cost_func, eta=0.1, epochs=3, mini_batch_size=None):
        self.eta = eta
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        self.a_funcs = [dummy_a] + act_funcs
        self.cost_func = cost_func

        self.num_layers = len(sizes)
        self.sizes = sizes

        self._biases = [0] + [np.random.randn(y, 1) for y in sizes[1:]]
        self._weights = [0] + [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        self.a = [0] + [np.zeros((y, 1)) for y in sizes[1:]]
        self.z = [np.zeros((y, 1)) for y in sizes]

        self.errors = [0] + [np.zeros((y, 1)) for y in sizes[1:]]

        self.scores = []

    @property
    def weights(self):
        return self._weights[1:]

    @property
    def biases(self):
        return self._biases[1:]

    def fit(self, x_train, y_train, x_test, y_test):
        if self.mini_batch_size:
            self._batch_fit(x_train, y_train, x_test, y_test)
        else:
            self._full_fit(x_train, y_train, x_test, y_test)

        return self

    def predict(self, x_pred):
        y_pred = np.array([self._feedforward(x).ravel() for x in x_pred])
        return y_pred

    def predict_proba(self, x_pred):
        y_pred = np.array([self._feedforward(x).ravel() for x in x_pred])
        return y_pred

    def evaluate(self, x_test, y_test):
        y_pred = [self._feedforward(x).ravel() for x in x_test]
        return self.cost_func.func(y_test, np.array(y_pred))

    #
    # Mode functions
    #

    def _sample_fit(self, x, y, n_samples):
        w_nabla, b_nabla = self._backprop(x, y)

        for l in range(self.num_layers)[1:]:
            self._weights[l] -= self.eta * w_nabla[l]/float(n_samples)
            self._biases[l] -= self.eta * b_nabla[l]/float(n_samples)
        return self

    def _full_fit(self, x_train, y_train, x_test, y_test):
        n_samples, _ = x_train.shape

        data_train = zip(x_train, y_train)
        for j in range(self.epochs):
            np.random.shuffle(data_train)
            for x, y in data_train:
                self._sample_fit(x, y, n_samples)

            self.scores.append(self.evaluate(x_test, y_test))
            # print "Epoch {0}: {1} ".format(j, self.scores[-1])

    def _batch_fit(self, x_train, y_train, x_test, y_test):
        n_samples, _ = x_train.shape

        data_train = zip(x_train, y_train)
        for j in xrange(self.epochs):
            np.random.shuffle(data_train)
            mini_batches = [
                data_train[k:k + self.mini_batch_size]
                for k in xrange(0, n_samples, self.mini_batch_size)]

            for mini_batch in mini_batches:
                for x, y in mini_batch:
                    self._sample_fit(x, y, self.eta)

                    # if test_data:
                    #     print "Epoch {0}: {1} / {2}".format(
                    #             j, self.evaluate(test_data), n_test)
                    # else:
                    #     print "Epoch {0} complete".format(j)

    #
    # Network Intrinsics
    #

    def _feedforward(self, x):
        x = x[:, np.newaxis]

        self.z[0] = x
        for l in range(self.num_layers)[1:]:
            self.a[l] = self._layer_a(l)
            self.z[l] = self.a_funcs[l].func(self.a[l])

        return self.z[-1]

    def _feedbackward(self, y):
        y = y[:, np.newaxis] if hasattr(y, '__len__') else y

        self.errors[-1] = self._output_error(y)

        for l in range(self.num_layers)[1:-1][::-1]:
            layer_error = self._layer_error(l)
            self.errors[l] = layer_error

    def _backprop(self, x, y):
        self._feedforward(x)
        self._feedbackward(y)

        w_nabla = [0]
        b_nabla = [0]

        for l in range(self.num_layers)[1:]:
            w_nabla.append(self._layer_w_nabla(l))
            b_nabla.append(self._layer_b_nabla(l))

        return w_nabla, b_nabla

    #
    # Formulas
    #

    def _output_error(self, y):
        return self.cost_func.derivative(y, self.z[-1]) * \
               self.a_funcs[-1].derivative(self.a[-1])

    def _layer_error(self, l):
        return np.dot(self._weights[l + 1].T, self.errors[l + 1]) * \
               self.a_funcs[l].derivative(self.a[l])

    def _layer_w_nabla(self, l):
        return np.dot(self.errors[l], self.z[l - 1].T)

    def _layer_b_nabla(self, l):
        return self.errors[l]

    def _layer_a(self, l):
        return np.dot(self._weights[l], self.z[l - 1]) + self._biases[l]

#
# _activation_funcs = {
#     'logistic': {
#         'function': activation_funcs.logistic_func,
#         'derivative': activation_funcs.logistic_derivative_func,
#     },
#     'tanh': {
#         'function': activation_funcs.tanh_func,
#         'derivative': activation_funcs.tanh_derivative_func,
#     },
#     'arctan': {
#         'function': activation_funcs.arctan_func,
#         'derivative': activation_funcs.arctan_derivative_func,
#     },
#     'softmax': {
#         'function': activation_funcs.softmax_func,
#         'derivative': activation_funcs.softmax_derivative_func,
#     },
# }
#
# _cost_funcs = {
#     'quadratic': {
#         'function': cost_funcs.quadratic_cost_func,
#         'derivative': cost_funcs.quadratic_cost_derivative_func,
#     },
#     'cross_entropy': {
#         'function': cost_funcs.cross_entropy_cost_func,
#         'derivative': cost_funcs.cross_entropy_cost_derivative_func,
#     },
#     'multiclass_cross_entropy': {
#         'function': cost_funcs.multiclass_cross_entropy_cost_func,
#         'derivative': cost_funcs.multiclass_cross_entropy_cost_derivative_func,
#     }
# }
