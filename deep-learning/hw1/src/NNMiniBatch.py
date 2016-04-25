# coding=utf-8
import sys

import numpy as np

from sklearn.preprocessing import normalize

from functions.cost_objs import *
from functions.activation_objs import *


# d - number of features (size of input layer)
# c - number of classes  (size of output layer)
# x = [       # y = [
#     [x1],   #     [y1],
#     [x2],   #     ...,
#     ...,    #     [yc]
#     [xd]    # ] (c x 1)
# ] (d x 1)

# L - number of layers
# z = [z0, z1, ..., zL]
# z0 = [
#     [x1],
#     [x1],
#     ...,
#     [xd],
# ] (d x 1) (N0 x 1)

# z1 = [h(a1)] (N1 x 1)
# zL = [h(aL)] (c = NL x 1)

# a = [a1, a2, ..., aL]
# a1 = [
#     [a1[0]  = np.dot(W1[0, :], z0)  + b1[0]],
#     [a1[1]  = np.dot(W1[1, :], z0)  + b1[1]],
#     ...,
#     [a1[N1] = np.dot(W1[N1, :], z0) + b1[N1]],
# ] (N1 x 1)   al =  np.dot(Wl, zl-1)  + bl

# W = [W1, W2, ..., WL]
# W1 (N1 x d)
# W2 (N2 x N1)

# W1 =[[ w0_0,   w0_1,   ...,  w0_d ],
#      [ w1_0,   w1_1,   ...,  w1_d ],
#      ...,
#      [ wN1_0,  wN1_1,  ...,  wN1_d],
# ] (N1 x N0)

# WL =[[ w0_0,     w0_1,     ...,  w0_NL-1 ],
#      [ w1_0,     w1_1,     ...,  w1_NL-1 ],
#      ...,
#      [ wNL_0,    wNL_1,    ...,  wNL_NL-1],
# ](NL x NL-1)

# b = [b1, b2, ..., bL]
# b1 = [
#     [b1[0]],
#     [b1[1]],
#     ...,
#     [b1[N1]]
# ] (N1 x 1)

# ξ - error
# ξ  = [ξ1, ξ2, ..., ξL]
# ξL = [
#     [ξL[0]  = h'(aL[0])  * E'(y, zL[0])],
#     [ξL[1]  = h'(aL[1])  * E'(y, zL[1])],
#     ...,
#     [ξL[NL] = h'(aL[NL]) * E'(y, zL[NL])],
# ] (NL x 1)   ξL =  h'(aL) * E'(y, zL)

# j - some hidden layer
# ξj = [
#     [ξj[0]  = h'(aj[0])   * np.dot(Wj+1[:, 0].T,  ξj+1) ],
#     [ξj[1]  = h'(aj[1])]  * np.dot(Wj+1[:, 1].T,  ξj+1) ],
#     ...,
#     [ξj[Nj] = h'(aj[Nj])] * np.dot(Wj+1[:, Nj].T, ξj+1) ],
# ] (NL x 1)   ξj =  h'(aj) * np.dot(Wj+1.T, ξj+1)


class NNMiniBatch:
    def __init__(self, sizes, activation_functions, cost_function,
                 epochs=1, eta=0.1, mini_batch_size=10, l1_rate=0.0, mode="batch", stop_rate=0.0,
                 isr=[1], r=0.05, betta=0.0, lyamda=0.1):
        self.eta = eta
        self.isr = isr
        self.r = r
        self.betta = betta
        self.lyamda = lyamda
        self.mode = mode
        self.stop_rate = stop_rate
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
        self.b = [[]] + [np.ones(s, dtype=np.float32) for s in b_sizes]

        self.a = [None] * self.L
        self.z = [None] * self.L
        self.ro = [0.0] * self.L
        self.err = [None] * self.L

        self.prev_score = 0

    def feedforward(self, x):
        """
        forward input propagation
        """
        self.z[0] = x

        for l in range(1, self.L):
            # a[l] = z[l-1] x W[l].T + b[l]
            self.a[l] = self.z[l - 1].dot(self.w[l].T) + self.b[l]

            # z[l] = h(a[l])
            self.z[l] = self.h[l](self.a[l])

        return self.z[-1]

    def backprop(self, x, y):
        """
        backpropagation algorithm
        """
        self.feedforward(x)

        nabla_w = [None] * self.L
        nabla_b = [None] * self.L
        ro = [0] * self.L

        # error backpropagation

        # output layer error
        # ξL =  h'(aL) * E'(y, zL)
        self.err[-1] = self.h_d[-1](self.a[-1]) * self.c_d(y, self.z[-1])

        for l in reversed(range(1, self.L - 1)):
            sparse_reg = 0.0
            if l in self.isr:
                ro = np.sum(self.z[l], axis=0) / np.float64(x.shape[0])
                sparse_reg = self.betta * (- self.r / ro + (1 - self.r) / (1 - ro))

            # tau =  ξl+1 x Wl+1
            tau_l = self.err[l + 1].dot(self.w[l + 1]) + sparse_reg

            # ξl = h'(al) * tau
            self.err[l] = self.h_d[l](self.a[l]) * tau_l

        for l in range(1, self.L):
            # ∇w_l = ξl.T x zl-1
            nabla_w[l] = self.err[l].T.dot(self.z[l - 1])

            # ∇b_l = I.T x ξl
            nabla_b[l] = np.ones((self.err[l].shape[0], 1), dtype=np.float64).T.dot(self.err[l])

        return nabla_w, nabla_b, ro

    def sgd(self, train_data, cv_data=None):
        """
        stochastic gradient descent
        """
        scores = []
        scores_diff = []

        for epoch in range(self.epochs):
            np.random.shuffle(train_data)  # inplace shuffle
            batch = train_data[:self.mini_batch_size]

            if self.mode == "batch":
                x, y = zip(*batch)
                x = np.array(x)
                y = np.array(y)

                nabla_w, nabla_b, ro = self.backprop(x, y)
                for l in range(1, self.L):
                    self.w[l] = (1 - self.eta * self.lyamda / x.shape[0]) * self.w[l] \
                                - self.eta * nabla_w[l] / np.float64(x.shape[0])
                    self.b[l] -= self.eta * nabla_b[l] / np.float64(x.shape[0])

            elif self.mode == "online":
                for x, y in batch:
                    x = x[:, np.newaxis].T
                    y = y[:, np.newaxis].T

                    nabla_w, nabla_b, ro = self.backprop(x, y)

                    for l in range(1, self.L):
                        if l in self.isr:
                            ro = ro[l]
                            sparse_reg = - self.r / ro + (1 - self.r) / (1 - ro)
                            self.ro[l] += np.sum(sparse_reg)

                        self.w[l] -= self.eta * nabla_w[l]
                        # (1.0 - self.eta * 1.0 / np.float32(batch_size)) * self.w[l] \
                        self.b[l] -= self.eta * nabla_b[l]

            if cv_data:
                matches, score = self.evaluate(cv_data)
                score_diff = self.prev_score - score

                scores.append(score)
                scores_diff.append(score_diff)

                if len(scores) > 10:
                    score_diff = np.mean(np.abs(scores_diff[epoch - 10:epoch]))

                sys.stdout.write(
                    '\r' + "Epoch {0}: {1} / {2} | {3} | {4}".format(epoch, matches,
                                                                     len(cv_data), score,
                                                                     score_diff)
                )
                sys.stdout.flush()

                self.prev_score = score

                if len(scores) > 10:
                    if abs(score_diff) < self.stop_rate:
                        break

        return scores_diff, scores

    def evaluate(self, cv_data):
        x, y = zip(*cv_data)
        x = np.array(x)
        y = np.array(y)

        y_pred = self.feedforward(x)

        matches = 0  # np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        score = self.c(y, y_pred)
        return matches, score


def net_test():
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

    train_data = zip(normalize(x_train.astype(np.float32)), y_train.astype(np.float32))

    nn = NNMiniBatch([2, 10, 2], [LogisticFunc, IdentyFunc], QuadraticCost,
                     epochs=100, mini_batch_size=2, eta=0.1)
    nn.sgd(train_data)

    x_test = np.array([
        [-10, 4],
        [-15, 2],
        [120, 1],
        [130, 5],
    ])

    test_data = normalize(x_test.astype(np.float32))

    print nn.feedforward(test_data)
    print np.argmax(nn.feedforward(test_data), axis=1)
    # nn.predict(test_data)

    # print nn.w


def autoencoder_test():
    x_train = np.array([
        [-10, 5],
        [-20, 4],
        [100, 7],
        [140, 5],
    ])

    train_data = zip(normalize(x_train.astype(np.float32)), normalize(x_train.astype(np.float32)))
    nn = NNMiniBatch([2, 10, 2], [LogisticFunc, IdentyFunc], QuadraticCost,
                     epochs=30, mini_batch_size=2, eta=0.04, mode="online")
    nn.sgd(train_data)

    x_test = np.array([
        [-10, 4],
        [-15, 2],
        [120, 1],
        [130, 5],
    ])

    test_data = normalize(x_test.astype(np.float32))

    print nn.feedforward(test_data)


if __name__ == '__main__':
    # net_test()
    autoencoder_test()
