# https://en.wikipedia.org/wiki/Activation_function

import numpy as np


def logistic_func(a):
    return 1.0 / (1.0 + np.exp(-a))


def logistic_derivative_func(a):
    return logistic_func(a) * (1 - logistic_func(a))


def tanh_func(a):
    return 2.0 / (1.0 + np.exp(-2 * a)) - 1


def tanh_derivative_func(a):
    return 1 - tanh_func(a) ** 2


def arctan_func(a):
    return np.arctan(a)


def arctan_derivative_func(a):
    return 1 / (a ** 2 + 1)


def softmax_func(a):
    s = np.exp(a)
    return s / np.sum(s)


def softmax_derivative_func(a):
    y = softmax_func(a)
    return y * (1 - y)
