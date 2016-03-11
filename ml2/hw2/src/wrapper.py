import time

import numpy as np
from sklearn.cross_validation import KFold

from joblib import Parallel, delayed

from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier

from constants import ADABOOST_PARAMS

FOLDS = 4


def fit_pred_abc(X, y, feature_set):
    score = 0
    abc = AdaBoostClassifier(*ADABOOST_PARAMS)

    for train_index, test_index in KFold(len(y), n_folds=FOLDS):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        abc.fit(X_train[:, feature_set], y_train)

        y_pred = abc.predict(X_test[:, feature_set])
        score += f1_score(y_test, y_pred) / float(FOLDS)

    return score


def wrapper(expected_features, x_train, y_train, x_test, y_test):
    initial_fset = set(np.arange(x_train.shape[1]))
    target_flist = list()
    times_list = []

    scores = []

    for i in range(expected_features):
        start = time.time()
        print i,
        target_fset = set(target_flist)
        rest_fset = list(initial_fset - target_fset)

        unbound_fsets = [list(target_fset | set([f])) for f in rest_fset]

        clfs_scores = Parallel(n_jobs=8)(delayed(fit_pred_abc)(x_train, y_train, fset) for fset in unbound_fsets)

        best_f_ind = np.argmax(clfs_scores)

        best_f = rest_fset[best_f_ind]
        target_flist.append(best_f)
        scores.append(clfs_scores[best_f_ind])
        end = time.time()
        times_list.append(end - start)

    return target_flist, scores, times_list


wrapper_features, wrapper_scores, wrapper_times = wrapper(2, x_train, y_train, x_test, y_test)
