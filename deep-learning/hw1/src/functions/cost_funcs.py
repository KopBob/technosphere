# http://stats.stackexchange.com/a/154880/87250

import numpy as np


class quadratic_cost:
    @staticmethod
    def function(y, z):
        return 0.5 * ((y - z) ** 2)

    @staticmethod
    def derivative(y, z):
        return z - y


class cross_entropy_cost:
    @staticmethod
    def function(y, z):
        return -np.sum(y * np.log(z) + (1 - y) * np.log(1 - z))

    @staticmethod
    def derivative(y, z):
        return (z - y) / ((z + 1) * z)


class multiclass_cross_entropy_cost:
    @staticmethod
    def function(y, z):
        return -np.sum(y * np.log(z))

    @staticmethod
    def derivative(y, z):
        return z - y
