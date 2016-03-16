import numpy as np


def weighted_majority(algos, x_train, y_train, betta=0.75):
    predictors = [alg.fit(x_train, y_train) for alg in algos]

    n_samples = x_train.shape[0]
    m_predictors = len(predictors)
    w = np.ones(m_predictors)

    prediction = np.zeros((n_samples, m_predictors))

    for n in range(n_samples):
        qneg = 0
        qpos = 0

        for m in range(m_predictors):
            pred = predictors[m].predict([x_train[n, :]])[0]

            qneg += w[m] if pred == -1 else 0
            qpos += w[m] if pred == 1 else 0

            if qneg > qpos:
                prediction[n, m] = -1
            elif qneg == qpos:
                prediction[n, m] = 1 if np.random.randint(0, 2) else -1
            else:
                prediction[n, m] = 1

            if prediction[n, m] != y_train[n]:
                w[m] *= betta
    return np.array(w)


def weighted_predictor(x, predictors, weights):
    pred = np.array([p.predict(x) for p in predictors])
    pos = np.sum(np.multiply(pred == 1, weights[:, np.newaxis]), axis=0)
    neg = np.sum(np.multiply(pred == -1, weights[:, np.newaxis]), axis=0)
    y_pred = np.argmax(np.vstack((neg, pos)), axis=0)
    y_pred[y_pred == 0] = -1
    return y_pred
