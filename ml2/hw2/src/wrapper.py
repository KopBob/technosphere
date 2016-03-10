import numpy as np

from sklearn.metrics import f1_score


def wrapper(expected_features, x_train, y_train, x_test, y_test):
    default_f_set = set(np.arange(x_train.shape[1]))
    f_set = set()
    p = abc(n_estimators=100, learning_rate=0.1)

    for _ in range(expected_features):
        preds = [(p.fit(x_train[:, list(f_set | set([f]))], y_train), f) for f in default_f_set - f_set]
        best_f = np.argmin([f1_score(y_test, p.predict(x_test[:, list(f_set | set([f]))])) for p, f in preds])
        f_set = f_set | set([best_f])

    return f_set
