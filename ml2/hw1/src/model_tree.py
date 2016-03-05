import numpy as np

from sklearn import cross_validation as cv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse


def best_split_lin_reg(x_vect, y):
    # cv_data = cv.train_test_split(x_vect, y, test_size=0.25, random_state=288)
    # x_train, x_test, y_train, y_test = cv_data

    node_lg = LinearRegression(n_jobs=4).fit(x_vect[:, np.newaxis], y)
    node_mse = mse(y, node_lg.predict(x_vect[:, np.newaxis]))

    best_score = -np.inf
    best_split_value = None
    best_true_inds = None
    best_false_inds = None

    for split_value in x_vect:
        true_inds = x_vect > split_value
        true_ratio = len(np.nonzero(true_inds)) / float(len(y))
        true_lg = LinearRegression(n_jobs=4).fit(x_vect[true_inds][:, np.newaxis], y[true_inds])
        true_score = mse(y, true_lg.predict(x_vect[:, np.newaxis]))

        false_inds = np.invert(true_inds)
        false_ratio = 1 - true_ratio
        false_lg = LinearRegression(n_jobs=4).fit(x_vect[false_inds][:, np.newaxis], y[false_inds])
        false_score = mse(y, false_lg.predict(x_vect[:, np.newaxis]))

        score = node_mse - (true_ratio * true_score + false_ratio * false_score)
        if score > best_score:
            best_score = score
            best_split_value = split_value
            best_true_inds = true_inds
            best_false_inds = false_inds

    return best_false_inds, best_true_inds, best_split_value, best_score


def best_split_mse_brute_force(x_vect, y):
    node_std = np.std(y) ** 2

    best_score = -np.inf
    best_split_value = None
    best_true_inds = None
    best_false_inds = None

    for split_value in x_vect:
        true_inds = x_vect >= split_value
        true_ratio = len(np.nonzero(true_inds)) / float(len(y))
        true_score = np.std(y[true_inds]) ** 2

        false_inds = np.invert(true_inds)
        false_ratio = 1 - true_ratio
        false_score = np.std(y[false_inds]) ** 2

        score = node_std - (true_ratio * true_score + false_ratio * false_score)
        if score > best_score:
            best_score = score
            best_split_value = split_value
            best_true_inds = true_inds
            best_false_inds = false_inds

    return best_false_inds, best_true_inds, best_split_value, best_score


def best_split_mse(x_vect, y):
    node_std = np.std(y) ** 2

    sorted_ind = np.argsort(x_vect)
    s_x = x_vect[sorted_ind]
    s_y = y[sorted_ind]

    left_sum = np.array([0] + list(np.cumsum(s_y)[:-1]))
    right_sum = np.abs(left_sum - left_sum[-1])

    left_len = np.arange(0, len(y), dtype=np.float64)
    left_ratio = left_len / float(len(y))
    right_len = np.arange(len(y), 0, -1, dtype=np.float64)
    right_ratio = right_len / float(len(y))

    left_crit = np.nan_to_num(left_ratio * ((left_sum - (left_sum ** 2) / left_len) / left_len))
    right_crit = np.nan_to_num(right_ratio * ((right_sum - (right_sum ** 2) / right_len) / right_len))

    score = node_std - (left_crit + right_crit)

    indices = range(len(s_x))

    for i in range(1, len(s_x)):
        if s_x[i - 1] == s_x[i]:
            indices[i] = indices[i - 1]

    sep_ind = np.argmax(score[indices])

    feature_value = s_x[sep_ind]
    false_objects = sorted_ind[:sep_ind]
    true_objects = sorted_ind[sep_ind:]
    best_score = score[sep_ind]

    return false_objects, true_objects, feature_value, best_score


def build_tree(x, y, depth=0, min_samples_leaf=100, max_depth=50):
    depth += 1
    if depth > max_depth:
        return Node(x, y)

    n_samples, m_features = x.shape

    splits = [(best_split_mse_brute_force(x[:, f_ind], y), f_ind) for f_ind in range(m_features)]
    splits_sorted = sorted(splits, key=lambda tup: tup[0][-1])

    (false_inds, true_inds, split_value, score), split_feature = splits_sorted[-1]
    false_x, false_y = x[false_inds], y[false_inds]
    true_x, true_y = x[true_inds], y[true_inds]

    size_cond = len(false_y) <= min_samples_leaf or len(true_y) <= min_samples_leaf

    if size_cond:
        return Node(x, y)

    false_branch = build_tree(false_x, false_y, depth, min_samples_leaf, max_depth)
    depth -= 1
    true_branch = build_tree(true_x, true_y, depth, min_samples_leaf, max_depth)
    depth -= 1

    return Node(x, y, split_feature=split_feature, split_value=split_value,
                false_branch=false_branch, true_branch=true_branch)


def tree_predict_one(x, tree):
    if tree.split_feature is None:
        return np.mean(tree.y)

    pred = x[tree.split_feature] >= tree.split_value
    branch = tree.tb if pred else tree.fb

    return tree_predict_one(x, branch)


def tree_predict(x, tree):
    return [tree_predict_one(obj, tree) for obj in x]


class Node(object):
    def __init__(self, x, y, split_feature=None, split_value=None, false_branch=None, true_branch=None):
        self.x = x
        self.y = y

        self.split_feature = split_feature
        self.split_value = split_value

        self.fb = false_branch
        self.tb = true_branch


class ModelTreeRegressor(object):
    def __init__(self):
        raise NotImplementedError

    def fit(self, x_train, y_train):
        raise NotImplementedError

    def predict(self, x_pred):
        raise NotImplementedError
