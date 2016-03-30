# https://en.wikipedia.org/wiki/Activation_function

import numpy as np


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


class softmax_activation:
    @staticmethod
    def function(a):
        s = np.exp(a)
        return s / np.sum(s)

    @staticmethod
    def derivative(a):
        y = softmax_activation.function(a)
        return y * (1 - y)
