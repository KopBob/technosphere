#!/usr/bin/env python
# -*- encoding: utf-8

import pandas as pd
import numpy as np
from sklearn import cross_validation as cv

from src.benchmark import plot_graph

from src.wrapper import wrapper


def run_wrapper(x_train, x_test, y_train, y_test):
    print "Running wrapper method..."
    wrapper_features, wrapper_scores, wrapper_times = wrapper(2, # x_train.shape[1] - 1,
                                                              x_train, y_train, x_test, y_test)

    plot_graph(wrapper_scores, "Wrapper Method Scores", "n_features", "f1-score",
               "./hw2_files/wrapper_method_scores.png")
    plot_graph(wrapper_times, "Wrapper Method Times", "n_features", "time",
               "./hw2_files/wrapper_method_times.png")

    print "Finished!"


if __name__ == '__main__':
    df = pd.read_csv("./spam.train.txt", delim_whitespace=True, header=None)
    x_data = df.ix[:, 1:].as_matrix();
    y_data = df.ix[:, 0].as_matrix()
    y_data[y_data == 0] = -1
    x_train, x_test, y_train, y_test = cv.train_test_split(x_data, y_data,
                                                           test_size=0.25, random_state=288)


    run_wrapper(x_train, x_test, y_train, y_test)
