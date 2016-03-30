# coding=utf-8
import sys

import numpy as np

from functions.activation_funcs import *
from functions.cost_funcs import *

from misc import convert2readable


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

        self.W = [None] + [np.random.normal(0, 0.1, s) for s in self.W_sizes]  # W = [None, W1, W2, ..., WL]
        self.b = [None] + [np.ones(s) for s in self.b_sizes]  # b = [None, b1, b2, ..., bL]
        # None is for nice indexing

        # L-1 activation functions
        self.a_funcs = [dummy_activation] + activation_funcs
        self.cost_func = cost

        self.z = [None] * self.L  # z = [z0, z1, ..., zL]
        self.a = [None] * self.L  # a = [None, a1, a2, ..., aL]
        self.err = [None] * self.L  # ξ  = [None, ξ1, ξ2, ..., ξL]

    def backprop(self, x, y):
        self.feedforward(x)  # calculate a, z
        self.feedbackward(y)  # calculate err

        w_nabla = [None] + [np.empty(s) for s in self.W_sizes]
        b_nabla = [None] + [np.empty(s) for s in self.b_sizes]

        for l in range(self.L)[1:]:  # from 1 to L, except 0 layer
            w_nabla[l] = np.dot(self.err[l], self.z[l - 1].T)  # Nl x Nl-1 = Nl x 1 * 1 x Nl-1
            b_nabla[l] = self.err[l]

        return w_nabla, b_nabla

    def feedforward(self, x):
        # a = [None, a1, a2, ..., aL]  al =  np.dot(Wl, zl-1)  + bl
        # z = [z0, z1, ..., zL]
        # add input layer
        self.z[0] = x
        for l in range(self.L)[1:]:  # from 1 to L, except 0 layer
            self.a[l] = np.dot(self.W[l], self.z[l - 1]) + self.b[l]  # (Nl x Nl-1 * Nl-1 x 1 + Nl x 1)
            self.z[l] = self.a_funcs[l].function(self.a[l])
        return self.z[-1]

    def feedbackward(self, y):
        # ξ  = [None, ξ1, ξ2, ..., ξL]
        # output layer error
        self.err[-1] = self.a_funcs[-1].derivative(self.a[-1]) * \
                       self.cost_func.derivative(y, self.z[-1])  # ξL =  h'(aL) * E'(y, zL)

        for l in reversed(range(self.L - 1)[1:]):  # from L-1 to 1, except 0 layer
            # ξj =  h'(aj) * np.dot(Wj+1.T, ξj+1)
            self.err[l] = self.a_funcs[l].derivative(self.a[l]) * \
                          np.dot(self.W[l + 1].T, self.err[l + 1])  # Nl x 1 * dot(Nl x Nl+1, Nl+1 * 1)

    def GD(self, train_data, cv_data=None):
        n_samples = len(train_data)

        for epoch in range(self.epochs):

            np.random.shuffle(train_data)
            for x, y in train_data:
                w_nabla, b_nabla = self.backprop(x, y)

                for l in range(self.L)[1:]:
                    if self.regularization is None:
                        self.W[l] -= (self.eta) * w_nabla[l]
                    elif self.regularization == 'l1':
                        self.W[l] = self.W[l] - self.eta * self.gamma / float(1) \
                                    - (self.eta / float(n_samples)) * w_nabla[l]
                    elif self.regularization == 'l2':
                        self.W[l] = self.W[l] * (1 - self.eta * self.gamma / float(1)) \
                                    - (self.eta / float(1)) * w_nabla[l]

                    self.b[l] -= (self.eta / float(1)) * b_nabla[l]

            if cv_data:
                sys.stdout.write('\r' + "Epoch {0}: {1} / {2}".format(epoch, self.evaluate(cv_data), len(cv_data)))
                sys.stdout.flush()

    def evaluate(self, cv_data):
        cv_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in cv_data]
        return sum(int(x == y) for (x, y) in cv_results)

    def predict(self, data):
        for x in data:
            print self.feedforward(x).ravel(), np.argmax(self.feedforward(x))


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
