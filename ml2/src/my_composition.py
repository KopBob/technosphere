import numpy as np

from sklearn.svm import LinearSVC, LinearSVR, SVR

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

svr_params = {
    "C": 300,
    "gamma": 0.6
}

svr_params2 = {
    "C": 0.1,
    "epsilon": 0.1
}


class MyComposition(object):
    def __init__(self, n_estimators=60):
        self.n_estimators = n_estimators
        self.lsvc = LinearSVC(penalty='l1', dual=False)

        self.estimators_ = None
        self.weights_ = None

    def fit(self, x_train, y_train, x_train2, y_train2):
        # svr = SVR(**svr_params)
        svr = LinearSVR(**svr_params2)
        gbr = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                        learning_rate=0.1)
        abr = AdaBoostRegressor(n_estimators=self.n_estimators,
                                learning_rate=0.01)

        estimators = [gbr, abr, svr]
        self.estimators_ = [e.fit(x_train, y_train) for e in estimators]

        x_pred = np.vstack([e.predict(x_train2) for e in self.estimators_]).T

        self.lsvc.fit(x_pred, y_train2)
        self.weights_ = np.array(self.lsvc.coef_ / np.sum(self.lsvc.coef_)).ravel()

        return self

    def predict(self, x):
        x_pred = np.vstack([e.predict(x) for e in self.estimators_]).T
        return np.sum(x_pred * self.weights_, axis=1).ravel()
