from joblib import Parallel, delayed

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingClassifier

from matplotlib import pyplot as plt

from GBoost import GBoost

MIN_SAMPLES_LEAF = 1
MAX_DEPTH = 4


def mse_gboost(x_train, x_test, y_train, y_test, n_estimators):
    estimators_range = range(1, n_estimators + 1)

    clf = GBoost(n_estimators=n_estimators + 1,
                 min_samples_leaf=MIN_SAMPLES_LEAF,
                 max_depth=MAX_DEPTH)
    clf.fit(x_train, y_train)

    return [mean_squared_error(y_test, clf.predict(x_test, n)) for n in estimators_range]


def mse_sklearn(x_train, x_test, y_train, y_test, n_estimators):
    clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                     min_samples_leaf=MIN_SAMPLES_LEAF,
                                     max_depth=MAX_DEPTH)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return mean_squared_error(y_test, pred)


def draw_graph_hw1(x_train, x_test, y_train, y_test, up_to_n_estimators=10):
    estimators_range = range(1, up_to_n_estimators + 1)
    gboost_scores = mse_gboost(x_train, x_test, y_train, y_test, up_to_n_estimators)

    sklearn_scores = Parallel(n_jobs=8)(delayed(mse_sklearn)(x_train, x_test, y_train, y_test, n) for n in estimators_range)
    # sklearn_scores = [mse_sklearn(x_train, x_test, y_train, y_test, n) for n in estimators_range]

    plt.plot(estimators_range, gboost_scores, color="red")
    plt.plot(estimators_range, sklearn_scores)

    return gboost_scores, sklearn_scores
