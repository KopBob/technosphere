import time

import numpy as np
from sklearn.cross_validation import KFold

from joblib import Parallel, delayed

from sklearn.metrics import f1_score, mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier

from constants import NUM_CORES

from my_composition import MyComposition

FOLDS = 4


def fit_pred_abc(X, y, feature_set):
    score = 0
    # abc = AdaBoostClassifier(**ADABOOST_PARAMS)
    abc = GradientBoostingClassifier(n_estimators=120, learning_rate=0.6)

    for train_index, test_index in KFold(len(y), n_folds=FOLDS):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        abc.fit(X_train[:, feature_set], y_train)

        y_pred = abc.predict(X_test[:, feature_set])
        score += f1_score(y_test, y_pred) / float(FOLDS)

    return score


svm_params = {'penalty': 'l1', 'loss': 'squared_hinge', 'C': 1, 'dual': False}


def fit_pred_composition(x_train, y_train, x_train2, y_train2, x_test, y_test, feature_set):
    score = 0

    # gbc = GradientBoostingClassifier(n_estimators=250)
    # svc = LinearSVC(**svm_params)
    # abc = AdaBoostClassifier(**ADABOOST_PARAMS)

    # predictors = [gbc, svc, abc]

    # for train_index, test_index in KFold(len(y_train), n_folds=FOLDS):
    #     x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
    #     y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
    #
    #     comp = MyComposition()
    #     comp.fit(x_train_cv[:, feature_set], y_train_cv, x_train2[:, feature_set], y_train2)
    #     y_pred = comp.predict(x_test_cv[:, feature_set])
    #
    #     # weights = weighted_majority(predictors, X_train[:, feature_set], y_train)
    #     # y_pred = weighted_predictor(X_test[:, feature_set], predictors, weights)
    #
    #     # score += f1_score(y_test, y_pred) / float(FOLDS)
    #     print x_test_cv[:, feature_set].shape, feature_set, y_pred, comp.weights_
    #     score += mean_squared_error(y_test_cv, y_pred) / float(FOLDS)

    comp = MyComposition()
    comp.fit(x_train[:, feature_set], y_train, x_train2[:, feature_set], y_train2)
    y_pred = comp.predict(x_test[:, feature_set])

    # weights = weighted_majority(predictors, X_train[:, feature_set], y_train)
    # y_pred = weighted_predictor(X_test[:, feature_set], predictors, weights)

    # score += f1_score(y_test, y_pred) / float(FOLDS)
    # print x_train[:, feature_set].shape, feature_set, y_pred, comp.weights_
    score = mean_squared_error(y_test, y_pred)

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

        # clfs_scores = Parallel(n_jobs=NUM_CORES)(
        #         delayed(fit_pred_composition)(x_train, y_train, x_train2, y_train2, x_test, y_test, fset) for fset in
        #         unbound_fsets)
        clfs_scores = Parallel(n_jobs=NUM_CORES)(
                delayed(fit_pred_abc)(x_train, y_train, fset) for fset in
                unbound_fsets)
        # clfs_scores = [fit_pred_composition(x_train, y_train, x_train2, y_train2, x_test, y_test, fset) for fset in unbound_fsets]

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
