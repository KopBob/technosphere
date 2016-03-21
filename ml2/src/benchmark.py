import os
import time
from sklearn.ensemble import GradientBoostingClassifier

from matplotlib import pyplot as plt
from joblib import Parallel, delayed

from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier

from GBoost import GBoost

from constants import NUM_CORES

MIN_SAMPLES_LEAF = 1
MAX_DEPTH = 4


def mse_gboost(x_train, x_test, y_train, y_test, n_estimators):
    estimators_range = range(1, n_estimators + 1)

    clf = GBoost(n_estimators=n_estimators + 1,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH)
    clf.fit(x_train, y_train)

    return [f1_score(y_test, clf.predict(x_test, n)) for n in estimators_range]


def gboost2scores(clf, x_test, y_test, n_estimators):
    estimators_range = range(1, n_estimators + 1)

    return [f1_score(y_test, clf.predict(x_test, n)) for n in estimators_range]


def mse_sklearn(x_train, x_test, y_train, y_test, n_estimators):
    clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                     min_samples_leaf=MIN_SAMPLES_LEAF,
                                     max_depth=MAX_DEPTH)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return f1_score(y_test, pred)


def draw_graph_hw1(x_train, x_test, y_train, y_test, up_to_n_estimators=10):
    estimators_range = range(1, up_to_n_estimators + 1)
    gboost_scores = mse_gboost(x_train, x_test, y_train, y_test, up_to_n_estimators)

    sklearn_scores = Parallel(n_jobs=8)(
            delayed(mse_sklearn)(x_train, x_test, y_train, y_test, n) for n in estimators_range)
    # sklearn_scores = [mse_sklearn(x_train, x_test, y_train, y_test, n) for n in estimators_range]

    plt.plot(estimators_range, gboost_scores, color="red")
    plt.plot(estimators_range, sklearn_scores)

    return gboost_scores, sklearn_scores


ada_best_params = {'n_estimators': 250, 'learning_rate': 1.0, 'algorithm': 'SAMME.R'}

FOLDS = 4


def fit_pred_abc(X, y, feature_set):
    start = time.time()
    score = 0
    # abc = AdaBoostClassifier(**ADABOOST_PARAMS)
    clf = GradientBoostingClassifier(n_estimators=120, learning_rate=0.6)

    for train_index, test_index in KFold(len(y), n_folds=FOLDS):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train[:, feature_set], y_train)

        y_pred = clf.predict(X_test[:, feature_set])
        score += f1_score(y_test, y_pred) / float(FOLDS)
    end = time.time()
    return clf, score, end - start


def predict_on_fsubset(features, x_train, x_test, y_train, y_test):
    start = time.time()
    # clf = AdaBoostClassifier(**ada_best_params)
    clf = GradientBoostingClassifier(n_estimators=120, learning_rate=0.6)
    # clf.fit(x_train[:, features], y_train)

    score = 0
    for train_index, test_index in KFold(len(y_train), n_folds=FOLDS):
        X_train, X_test = x_train[train_index], x_train[test_index]
        y_train, y_test = y_train[train_index], y_train[test_index]

        clf.fit(X_train[:, features], y_train)

        y_pred = clf.predict(X_test[:, features])
        score += f1_score(y_test, y_pred) / float(FOLDS)

    # y_pred = clf.predict(x_test[:, features])
    # score = f1_score(y_test, y_pred)

    end = time.time()
    return clf, score, end - start


def fset2scores(fset, x_train, x_test, y_train, y_test):
    fpacks = [fset[:i] for i in range(1, len(fset) + 1)]
    # res = Parallel(n_jobs=NUM_CORES)(delayed(predict_on_fsubset)(pack, x_train, x_test, y_train, y_test) for pack in fpacks)
    res = Parallel(n_jobs=NUM_CORES)(delayed(fit_pred_abc)(x_train, y_train, pack) for pack in fpacks)

    clfs, scores, times = zip(*res)
    return clfs, scores, times


def plot_graph(data, title, xlabel, ylabel, out_path=None):
    figure = plt.figure(figsize=(10, 10))
    plt.plot(range(1, len(data) + 1), data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if out_path:
        timestamp = time.strftime("%m-%d_%H-%M-%S")
        splited = os.path.splitext(out_path)
        new_path = "".join(list(splited)[:-1] + ["_", timestamp] + [splited[-1]])
        figure.savefig(new_path)
