# http://stats.stackexchange.com/a/154880/87250

import numpy as np


def quadratic_cost_func(y, z):
    return 0.5 * np.sum((z - y) ** 2)


def quadratic_cost_derivative_func(y, z):
    return z - y


def cross_entropy_cost_func(y, z):
    return -np.sum(y * np.log(z) + (1 - y) * np.log(1 - z))


def cross_entropy_cost_derivative_func(y, z):
    return (z - y) / ((z + 1) * z)


def multiclass_cross_entropy_cost_func(y, z):
    return -np.sum(y * np.log(z))


def multiclass_cross_entropy_cost_derivative_func(y, z):
    return z - y
