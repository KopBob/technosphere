# coding=utf-8
import numpy as np
from sklearn.preprocessing import normalize


class dummy_activation:
    @staticmethod
    def function(a):
        raise NotImplementedError

    @staticmethod
    def derivative(a):
        raise NotImplementedError


class logistic_activation:
    @staticmethod
    def function(a):
        return 1.0 / (1.0 + np.exp(-a * 1.0))

    @staticmethod
    def derivative(a):
        s = logistic_activation.function(a * 1.0)
        return 1.0 * s * (1 - s)


class identy_activation:
    @staticmethod
    def function(a):
        return a

    @staticmethod
    def derivative(a):
        return np.ones(a.shape)


class quadratic_cost:
    @staticmethod
    def function(y, z):
        return 0.5 * ((y - z) ** 2)

    @staticmethod
    def derivative(y, z):
        return z - y


class NN:
    def __init__(self, sizes, activation_funcs, cost,
                 regularization=None, eta=0.1, gamma=0.1, epochs=100):
        self.sizes = sizes
        self.activation_funcs = activation_funcs
        self.cost = cost
        self.regularization = regularization
        self.eta = eta
        self.gamma = gamma
        self.epochs = epochs

        self.L = len(sizes)

        self.W_sizes = zip(sizes[1:], sizes[:-1])
        self.b_sizes = zip(sizes[1:], np.tile(1, self.L - 1))

        print "W_sizes", self.W_sizes
        print "b_sizes", self.b_sizes

        self.W = [None] + [np.random.random(s) for s in self.W_sizes]  # W = [None, W1, W2, ..., WL]
        self.b = [None] + [np.ones(s) for s in self.b_sizes]  # b = [None, b1, b2, ..., bL]
        # None is for nice indexing

        # L-1 activation functions
        self.a_funcs = [dummy_activation] + activation_funcs
        self.cost_func = cost

        self.z = [None] * self.L  # z = [z0, z1, ..., zL]
        self.a = [None] * self.L  # a = [None, a1, a2, ..., aL]
        self.err = [None] * self.L  # ξ  = [None, ξ1, ξ2, ..., ξL]

    def backprop(self, x, y):
        try:
            if x.shape[1] != 1:
                raise BaseException("x invalid size, should be (d, 1)")
            if len(y) > 1 and y.shape[1] != 1:
                raise BaseException("y invalid size, should be (c, 1)")
        except IndexError:
            raise BaseException("x or y have invalid size", x.shape, y.shape)

        self.feedforward(x)  # calculate a, z
        self.feedbackward(y)  # calculate err

        w_nabla = [None] + [np.empty(s) for s in self.W_sizes]
        b_nabla = [None] + [np.empty(s) for s in self.b_sizes]

        # calculate nabla
        # from 1 to L, except 0 layer
        for l in range(self.L)[1:]:

            w_nabla[l] = np.dot(self.err[l], self.z[l - 1].T)  # Nl x Nl-1 = Nl x 1 * 1 x Nl-1
            if w_nabla[l].shape != self.W[l].shape:
                raise BaseException("w_nabla[l] invalid size, should be same as W[l]")

            b_nabla[l] = self.err[l]

        return w_nabla, b_nabla

    def feedforward(self, x):
        # z = [z0, z1, ..., zL]

        # add input layer
        self.z[0] = x

        if len(self.z) != self.L:
            raise BaseException("z invalid size, should be L")

        # a = [None, a1, a2, ..., aL]  al =  np.dot(Wl, zl-1)  + bl

        # from 1 to L, except 0 layer
        for l in range(self.L)[1:]:
            _a = np.dot(self.W[l], self.z[l - 1]) + self.b[l]  # (Nl x Nl-1 * Nl-1 x 1 + Nl x 1)
            if _a.shape[1] != 1:
                raise BaseException("a invalid size, should be (Nl, 1)")

            _z = self.a_funcs[l].function(_a)
            if _z.shape[1] != 1:
                raise BaseException("z invalid size, should be (Nl, 1)")

            self.a[l] = _a
            self.z[l] = _z

        return self.z[-1]

    def feedbackward(self, y):
        # ξ  = [None, ξ1, ξ2, ..., ξL]

        # output layer error
        self.err[-1] = self.a_funcs[-1].derivative(self.a[-1]) * \
                       self.cost_func.derivative(y, self.z[-1])  # ξL =  h'(aL) * E'(y, zL)
        if self.err[-1].shape[1] != 1:
            raise BaseException("err[L] invalid size, should be (NL, 1)")

        # from L-1 to 1, except 0 layer
        for l in reversed(range(self.L - 1)[1:]):
            # ξj =  h'(aj) * np.dot(Wj+1.T, ξj+1)
            _err = self.a_funcs[l].derivative(self.a[l]) * \
                   np.dot(self.W[l + 1].T, self.err[l + 1])  # Nl x 1 * dot(Nl x Nl+1, Nl+1 * 1)
            if _err.shape[1] != 1:
                raise BaseException("_err invalid size, should be (Nl, 1)")
            self.err[l] = _err

    def GD(self, data):
        n_samples = len(data)

        for epoch in range(self.epochs):

            np.random.shuffle(data)
            for x, y in data:
                w_nabla, b_nabla = self.backprop(x, y)

                for l in range(self.L)[1:]:
                    if self.regularization is None:
                        W_new = self.W[l] \
                                - (self.eta / float(n_samples)) * w_nabla[l]
                        self.W[l] = W_new
                    elif self.regularization == 'l1':
                        self.W[l] = self.W[l] - self.eta * self.gamma / float(n_samples) \
                                    - (self.eta / float(n_samples)) * w_nabla[l]
                    elif self.regularization == 'l2':
                        self.W[l] = self.W[l] * (1 - self.eta * self.gamma / float(n_samples)) \
                                    - (self.eta / float(n_samples)) * w_nabla[l]

                    b_new = self.b[l] - (self.eta / float(n_samples)) * b_nabla[l]
                    self.b[l] = b_new

    def predict(self, data):
        for x in data:
            print self.feedforward(x).ravel(), np.argmax(self.feedforward(x))


def convert2readable(x, norm=False):
    if norm:
        x = normalize(x.astype(np.float64))

    if len(x.shape) > 1:
        n_samples, d_features = x.shape
        _x = x.reshape((n_samples, d_features, 1)).astype(np.float64)
    else:
        n_samples = x.shape[0]
        _x = x.reshape((n_samples, 1)).astype(np.float64)

    return _x


if __name__ == '__main__':
    x_train = np.array([
        [-10, 5],
        [-20, 4],
        [100, 7],
        [140, 5],
    ])

    y_train = np.array([
        10,  # [0, 1],
        9,  # [0, 1],
        203,  # [1, 0],
        206,  # [1, 0],
    ])

    y_train = np.array([
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
    ])

    train_data = zip(convert2readable(x_train, norm=True), convert2readable(y_train))

    nn = NN([2, 3, 2], [logistic_activation, identy_activation], quadratic_cost)
    nn.GD(train_data)

    x_test = np.array([
        [-10, 4],
        [-15, 2],
        [120, 1],
        [130, 5],
    ])

    test_data = convert2readable(x_test, norm=True)

    nn.predict(test_data)
