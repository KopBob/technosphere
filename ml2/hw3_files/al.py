import argparse
import commands
import numpy as np
import matplotlib.pyplot as pl

from math import sqrt

# Import the linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

__author__ = 'gulinvladimir'

def main():
    args = parse_args()

    train_data = np.loadtxt(args.train)
    test_data  = np.loadtxt(args.test)

    total_data = np.concatenate(([train_data, test_data]), axis=0)

    number_of_features = len(train_data[0, :])
    number_of_items = len(train_data)

    #Linear regression
    print "Build linear regression..."
    linear_regression = LinearRegression()
    linear_regression.fit(train_data[0::, 1::], train_data[0::, 0])
    lr_prediction = linear_regression.predict(test_data[0::, 1::])

    rmse = sqrt(mean_squared_error(test_data[0::, 0], lr_prediction))

    print 'Rmse = ' + str(rmse)

    #Example oracle call
    print oracle(1, 1, 1, 0.8, 0.8, 1, 1, 0.8, 1, 0.8)

def oracle(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    query = "java -cp OracleRegression.jar Oracle " + str(x1) + " " + str(x2) + " " + str(x3) + " " + \
            str(x4) + " " + str(x5) + " " + " " + str(x6) + " " + str(x7) + " " + \
            str(x8) + " " + str(x9) + " " + str(x10)

    return commands.getoutput(query)


def parse_args():
    parser = argparse.ArgumentParser(description='Active Learning Tutorial')
    parser.add_argument("-tr", "--train", action="store", type=str, help="Train file name")
    parser.add_argument("-te", "--test", action="store", type=str, help="Test file name")
    return parser.parse_args()

if __name__ == "__main__":
    main()
