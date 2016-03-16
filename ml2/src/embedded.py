def embedded(tree, feature_weigths):
    if tree.split_feature is None:
        return feature_weigths

    feature_weigths[tree.split_feature] += tree.score

    feature_weigths = embedded(tree.fb, feature_weigths)
    feature_weigths = embedded(tree.tb, feature_weigths)

    return feature_weigths

# embedded_features = np.argsort(np.sum(
#     np.array([wide_walk(p.tree, np.zeros(x_train.shape[1])) for p in predictors[0].ensembles]),
#     axis=0))[::-1]