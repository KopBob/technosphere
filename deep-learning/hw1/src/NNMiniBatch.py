# coding=utf-8
import sys

import numpy as np

from sklearn.preprocessing import normalize

from functions.cost_objs import *
from functions.activation_objs import *


class NNMiniBatch:
    def __init__(self, sizes, activation_functions, cost_function,
                 epochs=1, eta=0.1, mini_batch_size=10, l1_rate=0.1):
        self.eta = eta
        self.l1_rate = l1_rate
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        self.h = [DummyFunc.function] + \
                 [f.function for f in activation_functions]
        self.h_d = [DummyFunc.function] + \
                   [f.derivative for f in activation_functions]

        self.c = cost_function.function
        self.c_d = cost_function.derivative

        self.L = len(sizes)

        # [3, 2, 3]
        # w_sizes - [(2, 3), (3, 2)]
        # b_sizes - [(1, 2), (1, 3)]
        w_sizes = zip(sizes[1:], sizes[:-1])
        b_sizes = zip(np.tile(1, len(sizes) - 1), sizes[1:])

        print "w_sizes", w_sizes
        print "b_sizes", b_sizes

        self.w = [[]] + [np.random.normal(0, 0.1, s) for s in w_sizes]
        # self.w = [[]] + [np.zeros(s) for s in w_sizes]
        self.b = [[]] + [np.ones(s, dtype=np.float64) for s in b_sizes]

        self.a = [None] * self.L
        self.z = [None] * self.L
        self.err = [None] * self.L

    def feedforward(self, x):
        self.z[0] = x

        for l in range(1, self.L):
            self.a[l] = self.z[l - 1].dot(self.w[l].T) + self.b[l]
            self.z[l] = self.h[l](self.a[l])

        return self.z[-1]

    def backprop(self, x, y):
        self.feedforward(x)

        nabla_w = [None] * self.L
        nabla_b = [None] * self.L

        self.err[-1] = self.h_d[-1](self.a[-1]) * self.c_d(y, self.z[-1])

        for l in reversed(range(1, self.L - 1)):
            tau_l = self.err[l + 1].dot(self.w[l + 1])
            self.err[l] = self.h_d[l](self.a[l]) * tau_l

        for l in range(1, self.L):
            nabla_w[l] = self.err[l].T.dot(self.z[l - 1])
            # nabla_b[l] = self.err[l][0]
            # nabla_b[l] = np.sum(self.err[l], axis=0)
            nabla_b[l] = np.ones((self.err[l].shape[0], 1), dtype=np.float64).T.dot(self.err[l])
        # print nabla_b[1]
        # print "\n", np.sum(self.b[1])
        return nabla_w, nabla_b

    def sgd(self, train_data, cv_data=None):
        for epoch in range(self.epochs):
            np.random.shuffle(train_data)  # inplace shuffle

            batch = train_data[:self.mini_batch_size]
            x, y = zip(*batch)
            x = np.array(x)
            y = np.array(y)

            nabla_w, nabla_b = self.backprop(x, y)

            for l in range(1, self.L):
                self.w[l] -= self.eta * nabla_w[l] / np.float64(len(batch)) \
                             - self.eta * self.l1_rate / np.float64(len(batch))
                self.b[l] -= self.eta * nabla_b[l] / np.float64(len(batch)) \
                             - self.eta * self.l1_rate / np.float64(len(batch))

            if cv_data:
                sys.stdout.write(
                        '\r' + "Epoch {0}: {1} / {2} | | {3}".format(epoch,
                                                                   self.evaluate(cv_data),
                                                                   len(cv_data),
                                                                   y.shape[0]
                                                                   )
                )
                sys.stdout.flush()

    def evaluate(self, cv_data):
        x, y = zip(*cv_data)

        # y_pred = self.feedforward(np.array(x))
        # y_true = y
        #
        # return self.c(y_true, y_pred)

        y_pred = np.argmax(self.feedforward(np.array(x)), axis=1)
        y_true = np.argmax(y, axis=1)
        return np.sum(y_pred == y_true)


if __name__ == '__main__':
    x_train = np.array([
        [-10, 5],
        [-20, 4],
        [100, 7],
        [140, 5],
    ])

    # y_train = np.array([
    #     [10],  # [0, 1],
    #     [9],  # [0, 1],
    #     [203],  # [1, 0],
    #     [206],  # [1, 0],
    # ])

    y_train = np.array([
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
    ])

    train_data = zip(normalize(x_train.astype(np.float64)), y_train.astype(np.float64))

    nn = NNMiniBatch([2, 10, 2], [LogisticFunc, IdentyFunc], QuadraticCost,
                     epochs=100, mini_batch_size=2, eta=0.1)
    nn.sgd(train_data)

    x_test = np.array([
        [-10, 4],
        [-15, 2],
        [120, 1],
        [130, 5],
    ])

    test_data = normalize(x_test.astype(np.float64))

    print nn.feedforward(test_data)
    print np.argmax(nn.feedforward(test_data), axis=1)
    # nn.predict(test_data)

    # print nn.w
