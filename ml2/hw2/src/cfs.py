import numpy as np


def cfs(x_train, y_train, expected_features):
    yX = np.vstack((y_train, x_train.T)).T
    target_corr = np.abs(np.corrcoef(yX.T)[0, :][1:])
    feature_corr = np.abs(np.corrcoef(x_train.T))

    feature_set = []

    numerator = 0
    denominator = 0
    total_score = -np.inf

    for _ in range(expected_features):
        best_score = total_score
        best_pos = -1
        best_feature_numerator_invest = 0
        best_feature_denominator_invest = 0

        for feature in range(x_train.shape[1]):
            if feature not in feature_set:
                numerator_invest = target_corr[feature]
                denominator_invest = 1 + 2 * np.sum(feature_corr[feature, feature_set])

                tmp_numerator = numerator + numerator_invest
                tmp_denominator = denominator + denominator_invest

                tmp_score = tmp_numerator / np.sqrt(tmp_denominator)

                if tmp_score > best_score:
                    best_feature_numerator_invest = numerator_invest
                    best_feature_denominator_invest = denominator_invest
                    best_pos = feature
                    best_score = tmp_score

        if best_pos == -1:
            continue

        numerator += best_feature_numerator_invest
        denominator += best_feature_denominator_invest
        total_score = numerator / np.sqrt(denominator)
        feature_set.append(best_pos)

    return feature_set
