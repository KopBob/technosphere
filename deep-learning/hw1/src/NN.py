# coding=utf-8

import numpy as np

from sklearn.preprocessing import normalize


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




class dummy_activation:
    @staticmethod
    def function(a):
        raise NotImplementedError

    @staticmethod
    def derivative(a):
        raise NotImplementedError


class dummy_cost:
    @staticmethod
    def function(y, z):
        raise NotImplementedError

    @staticmethod
    def derivative(y, z):
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


sizes = [2, 2, 1]
print "sizes", sizes
L = len(sizes)

# W = [(2,3), (3,2)] = zip(sizes[1:], sizes[:-1])
# b = [(2, 1), (3,1)] = zip(sizes[1:], np.tile(1, L-1))

W_sizes = zip(sizes[1:], sizes[:-1])
b_sizes = zip(sizes[1:], np.tile(1, L - 1))

print "W_sizes", W_sizes
print "b_sizes", b_sizes

W = [None] + [np.random.random(s) for s in W_sizes]  # W = [None, W1, W2, ..., WL]
b = [None] + [np.random.random(s) for s in b_sizes]  # b = [None, b1, b2, ..., bL]
# None is for nice indexing

# L-1 activation functions
a_funcs = [dummy_activation, logistic_activation, identy_activation]
cost_func = quadratic_cost

z = [None] * L  # z = [z0, z1, ..., zL]
a = [None] * L  # a = [None, a1, a2, ..., aL]
err = [None] * L  # ξ  = [None, ξ1, ξ2, ..., ξL]


def backprop(x, y):
    try:
        if x.shape[1] != 1:
            raise BaseException("x invalid size, should be (d, 1)")
        if len(y) > 1 and y.shape[1] != 1:
            raise BaseException("y invalid size, should be (c, 1)")
    except IndexError:
        raise BaseException("x or y have invalid size", x.shape, y.shape)

    feedforward(x)  # calculate a, z
    feedbackward(y)  # calculate err

    w_nabla = [None] + [np.empty(s) for s in W_sizes]
    b_nabla = [None] + [np.empty(s) for s in b_sizes]

    # calculate nabla
    # from 1 to L, except 0 layer
    for l in range(L)[1:]:
        print " l=", l

        w_nabla[l] = np.dot(err[l], z[l - 1].T)  # Nl x Nl-1 = Nl x 1 * 1 x Nl-1
        print "  w_nabla", w_nabla[l], z[l - 1].T
        if w_nabla[l].shape != W[l].shape:
            raise BaseException("w_nabla[l] invalid size, should be same as W[l]")

        b_nabla[l] = err[l]
        print "  b_nabla", b_nabla[l]

    return w_nabla, b_nabla


def feedforward(x):
    # z = [z0, z1, ..., zL]

    # add input layer
    z[0] = x

    if len(z) != L:
        raise BaseException("z invalid size, should be L")

    # a = [None, a1, a2, ..., aL]  al =  np.dot(Wl, zl-1)  + bl

    # from 1 to L, except 0 layer
    for l in range(L)[1:]:
        _a = np.dot(W[l], z[l - 1]) + b[l]  # (Nl x Nl-1 * Nl-1 x 1 + Nl x 1)
        if _a.shape[1] != 1:
            raise BaseException("a invalid size, should be (Nl, 1)")

        _z = a_funcs[l].function(_a)
        if _z.shape[1] != 1:
            raise BaseException("z invalid size, should be (Nl, 1)")

        a[l] = _a
        z[l] = _z

    return z[-1]


def feedbackward(y):
    # ξ  = [None, ξ1, ξ2, ..., ξL]

    # output layer error
    err[-1] = a_funcs[-1].derivative(a[-1]) * \
              cost_func.derivative(y, z[-1])  # ξL =  h'(aL) * E'(y, zL)
    if err[-1].shape[1] != 1:
        raise BaseException("err[L] invalid size, should be (NL, 1)")
    print "err", err[-1], z[-1], y

    # from L-1 to 1, except 0 layer
    for l in reversed(range(L - 1)[1:]):
        # ξj =  h'(aj) * np.dot(Wj+1.T, ξj+1)
        _err = a_funcs[l].derivative(a[l]) * \
               np.dot(W[l + 1].T, err[l + 1])  # Nl x 1 * dot(Nl x Nl+1, Nl+1 * 1)
        if _err.shape[1] != 1:
            raise BaseException("_err invalid size, should be (Nl, 1)")
        print "  l=", l, " ", _err
        err[l] = _err


def GD(data, regulariztion=None, eta=0.1, gamma=0.1, epochs=100):
    n_samples = len(data)

    for epoch in range(epochs):
        print epoch

        # np.random.shuffle(data)
        for x, y in data:
            w_nabla, b_nabla = backprop(x, y)

            for l in range(L)[1:]:
                if regulariztion is None:
                    W_new = W[l] - (eta / float(n_samples)) * w_nabla[l]
                    print "  W_new", W_new
                    print "  W_new - W_old", W_new - W[l]
                    W[l] = W_new
                elif regulariztion == 'l1':
                    W[l] = W[l] - eta * gamma / float(n_samples) - (eta / float(n_samples)) * w_nabla[l]
                elif regulariztion == 'l2':
                    W[l] = W[l] * (1 - eta * gamma / float(n_samples)) - (eta / float(n_samples)) * w_nabla[l]

                b_new = b[l] - (eta / float(n_samples)) * b_nabla[l]
                print "  B_new", b_new
                print "  B_new - B_old", b_new - b[l]
                b[l] = b_new



def predict(data):
    for x in data:
        print feedforward(x), np.argmax(feedforward(x))


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
        10,# [0, 1],
        9,# [0, 1],
        203, # [1, 0],
        206, # [1, 0],
    ])

    train_data = zip(convert2readable(x_train, norm=True), convert2readable(y_train))
    GD(train_data, epochs=10, regulariztion='l2')

    x_test = np.array([
        [-10, 4],
        [-15, 2],
        [120, 1],
        [130, 5],
    ])

    test_data = convert2readable(x_test, norm=True)

    predict(test_data)
