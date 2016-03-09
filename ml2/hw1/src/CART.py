import numpy as np


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

    _, unique_ind, unique_count = np.unique(s_x, return_index=True, return_counts=True)
    indices = np.repeat(unique_ind, unique_count)

    # indices = range(len(s_x))
    # for i in range(1, len(s_x)):
    #     if s_x[i - 1] == s_x[i]:
    #         indices[i] = indices[i - 1]

    sep_ind = np.argmax(score[indices])

    feature_value = s_x[sep_ind]
    false_objects = sorted_ind[:sep_ind]
    true_objects = sorted_ind[sep_ind:]
    best_score = score[sep_ind]

    return false_objects, true_objects, feature_value, best_score


def tree_predict_one(x, tree):
    if tree.split_feature is None:
        return np.mean(tree.y)*tree.gamma

    pred = x[tree.split_feature] >= tree.split_value
    branch = tree.tb if pred else tree.fb

    return tree_predict_one(x, branch)


class Node(object):
    def __init__(self, x, y, split_feature=None, split_value=None,
                 false_branch=None, true_branch=None, score=None, shrinkage=False):
        self.x = x
        self.y = y

        self.split_feature = split_feature
        self.split_value = split_value

        self.fb = false_branch
        self.tb = true_branch

        self.score = score

        self.gamma = 1

        if (split_value is None) and shrinkage:
            self.gamma = (np.sum(self.x)/(np.sum(np.abs(self.x) * (2 - np.abs(self.x)))))
            # print self.gamma


class CART(object):
    def __init__(self, min_samples_leaf=1, max_depth=4, shrinkage=False):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.shrinkage = shrinkage

        self.tree = None

    def build_tree(self, x, y, depth=0):
        depth += 1
        if depth > self.max_depth:
            return Node(x, y, shrinkage=self.shrinkage)

        n_samples, m_features = x.shape

        splits = [(best_split_mse(x[:, f_ind], y), f_ind) for f_ind in range(m_features)]
        splits_sorted = sorted(splits, key=lambda tup: tup[0][-1])

        (false_inds, true_inds, split_value, score), split_feature = splits_sorted[-1]

        false_x, false_y = x[false_inds], y[false_inds]
        true_x, true_y = x[true_inds], y[true_inds]

        size_cond = len(false_y) <= self.min_samples_leaf or len(true_y) <= self.min_samples_leaf

        if size_cond:
            return Node(x, y, shrinkage=self.shrinkage)

        false_branch = self.build_tree(false_x, false_y, depth)
        depth -= 1
        true_branch = self.build_tree(true_x, true_y, depth)
        depth -= 1

        return Node(x, y, score=score,
                    split_feature=split_feature, split_value=split_value,
                    false_branch=false_branch, true_branch=true_branch, shrinkage=self.shrinkage)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        return self

    def predict(self, x):
        return [tree_predict_one(obj, self.tree) for obj in x]
