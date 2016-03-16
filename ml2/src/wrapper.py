import time

import numpy as np
from sklearn.cross_validation import KFold

from joblib import Parallel, delayed

from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

from constants import ADABOOST_PARAMS, NUM_CORES
from weighted_majority import weighted_majority, weighted_predictor

FOLDS = 3


def fit_pred_abc(X, y, feature_set):
    score = 0
    # abc = AdaBoostClassifier(**ADABOOST_PARAMS)
    abc = GradientBoostingClassifier(n_estimators=120)

    for train_index, test_index in KFold(len(y), n_folds=FOLDS):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        abc.fit(X_train[:, feature_set], y_train)

        y_pred = abc.predict(X_test[:, feature_set])
        score += f1_score(y_test, y_pred) / float(FOLDS)

    return score


svm_params = {'penalty': 'l1', 'loss': 'squared_hinge', 'C': 1, 'dual': False}


def fit_pred_composition(X, y, feature_set):
    score = 0

    gbc = GradientBoostingClassifier(n_estimators=250)
    svc = LinearSVC(**svm_params)
    abc = AdaBoostClassifier(**ADABOOST_PARAMS)

    predictors = [gbc, svc, abc]

    for train_index, test_index in KFold(len(y), n_folds=FOLDS):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        weights = weighted_majority(predictors, X_train[:, feature_set], y_train)
        y_pred = weighted_predictor(X_test[:, feature_set], predictors, weights)

        score += f1_score(y_test, y_pred) / float(FOLDS)

    return score


def wrapper(expected_features, x_train, y_train, x_test, y_test):
    initial_fset = set(np.arange(x_train.shape[1]))
    target_flist = list()
    times_list = []

    scores = []

    for i in range(expected_features):
        start = time.time()
        target_fset = set(target_flist)
        rest_fset = list(initial_fset - target_fset)

        unbound_fsets = [list(target_fset | set([f])) for f in rest_fset]

        clfs_scores = Parallel(n_jobs=NUM_CORES)(
                delayed(fit_pred_abc)(x_train, y_train, fset) for fset in unbound_fsets)
        # clfs_scores = [fit_pred_abc(x_train, y_train, fset) for fset in unbound_fsets]

        best_f_ind = np.argmax(clfs_scores)

        best_f = rest_fset[best_f_ind]
        target_flist.append(best_f)
        scores.append(clfs_scores[best_f_ind])
        end = time.time()
        times_list.append(end - start)
        print i, target_flist
        print i, scores
        print i, times_list

    return target_flist, scores, times_list

# wrapper_features, wrapper_scores, wrapper_times = wrapper(2, x_train, y_train, x_test, y_test)
