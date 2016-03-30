import numpy as np


class DummyFunc:
    @staticmethod
    def function(*args):
        raise NotImplementedError

    @staticmethod
    def derivative(*args):
        raise NotImplementedError


class LogisticFunc:
    @staticmethod
    # @np.vectorize
    def function(a):
        return 1.0 / (1.0 + np.exp(-a))

    @staticmethod
    # @np.vectorize
    def derivative(a):
        s = LogisticFunc.function(a)
        return s * (1.0 - s)


class IdentyFunc:
    @staticmethod
    # @np.vectorize
    def function(a):
        return a

    @staticmethod
    def derivative(a):
        return 1.0  # np.ones(a.shape, dtype=np.float64)


class SoftmaxFunc:
    @staticmethod
    def function(a):
        s = np.exp(a)
        return (s.T / np.sum(s, axis=1)).T

    @staticmethod
    def derivative(a):
        y = SoftmaxFunc.function(a)
        return y * (1 - y)


class TanhFunc:
    @staticmethod
    # @np.vectorize
    def function(a, r=1.0):
        return np.tanh(a * r)

    @staticmethod
    # @np.vectorize
    def derivative(a, r=1.0):
        t = np.tanh(r * a)
        return r * (1 - t * t)


class ReLuFunc:
    @staticmethod
    def function(a):
        return a * (a > 0).astype(np.float64)

    @staticmethod
    # @np.vectorize
    def derivative(a):
        return (a > 0).astype(np.float64)
