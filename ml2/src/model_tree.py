import numpy as np

from sklearn import cross_validation as cv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

from constants import NUM_CORES

def ling_reg_score(indices, x, y):
    if not np.any(indices):
        return 1
    leaf_x = x[indices][:, np.newaxis]
    leaf_y = y[indices]
    lg = LinearRegression(n_jobs=NUM_CORES).fit(leaf_x, leaf_y)
    score = mse(leaf_y, lg.predict(leaf_x))
    return score


def best_split_lin_reg(x_vect, y):
    node_lg = LinearRegression(n_jobs=NUM_CORES).fit(x_vect[:, np.newaxis], y)
    node_score = mse(y, node_lg.predict(x_vect[:, np.newaxis]))

    best_score = -np.inf
    best_split_value = None
    best_true_inds = None
    best_false_inds = None

    for split_value in np.unique(x_vect):
        true_inds = x_vect > split_value
        true_ratio = np.sum(true_inds) / float(len(y))
        true_score = ling_reg_score(true_inds, x_vect, y)

        false_inds = np.invert(true_inds)
        false_ratio = 1 - true_ratio
        false_score = ling_reg_score(false_inds, x_vect, y)

        score = node_score - (true_ratio * true_score + false_ratio * false_score)

        if score > best_score:
            best_score = score
            best_split_value = split_value
            best_true_inds = true_inds
            best_false_inds = false_inds

    return best_false_inds, best_true_inds, best_split_value, best_score


def best_split_lin_reg_dynamic(x, y):
    sort_i = np.argsort(x)

    n = len(y)

    xm = x[sort_i] - np.mean(x)
    ym = y[sort_i] - np.mean(y)

    xy_sum_false = 0
    x2_sum_false = 0
    xy_sum_true = np.sum(xm * ym)
    x2_sum_true = np.sum(xm ** 2)

    node_betta = xy_sum_true / x2_sum_true
    node_betta = 1 if np.isnan(node_betta) else node_betta
    node_score = mse(ym + np.mean(y), node_betta * xm)

    best_score = np.inf
    split_value = xm[0] + np.mean(x)
    split_ind = x[sort_i[0]]

    for i in range(1, n):
        xy_sum_false += xm[i] * ym[i]
        x2_sum_false += xm[i] ** 2

        xy_sum_true -= xm[i] * ym[i]
        x2_sum_true -= xm[i] ** 2

        print xm[i] + np.mean(x)

        false_betta = xy_sum_false / x2_sum_false
        # false_betta = 0 if np.isnan(false_betta) else false_betta
        false_ratio = i / float(n)
        false_score = mse(ym[:i] + np.mean(y), false_betta * xm[:i]) if len(ym[:i]) else 0

        true_betta = xy_sum_true / x2_sum_true
        # true_betta = 0 if np.isnan(true_betta) else true_betta
        true_ratio = (n - i) / float(n)
        true_score = mse(ym[i:] + np.mean(y), true_betta * xm[i:])

        score = node_score - (false_ratio * false_score + true_ratio * true_score)
        scores = np.array([false_score, true_score])
        score = scores[np.argmin(scores)]

        if score < best_score:
            best_score = score
            split_value = x[sort_i[i]]
            split_ind = i

    return sort_i[:split_ind], sort_i[split_ind:], split_value, best_score


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

    splits = [(best_split_mse(x[:, f_ind], y), f_ind) for f_ind in range(m_features)]
    splits_sorted = sorted(splits, key=lambda tup: tup[0][-1])
    # print [s[0][-1] for s in splits_sorted]

    (false_inds, true_inds, split_value, score), split_feature = splits_sorted[-1]
    false_x, false_y = x[false_inds], y[false_inds]
    true_x, true_y = x[true_inds], y[true_inds]

    # print len(false_y), len(true_y)

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
        # return tree.lg.predict(x)
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

        self.lg = None

        if split_feature is None:
            self.lg = LinearRegression().fit(self.x, self.y)


class ModelTreeRegressor(object):
    def __init__(self):
        raise NotImplementedError

    def fit(self, x_train, y_train):
        raise NotImplementedError

    def predict(self, x_pred):
        raise NotImplementedError
