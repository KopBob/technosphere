import numpy as np

from CART import CART, tree_predict_one


def logic_loss(y_true, y_pred):
    return 2 * y_true / (1 + np.exp(2 * y_true * y_pred))


pos = lambda y: 1 / (1 + np.exp(-2 * y))
neg = lambda y: 1 / (1 + np.exp(2 * y))


class GBoost:
    def __init__(self, n_estimators, min_samples_leaf=1,
                 max_depth=4, learning_rate=1, loss=logic_loss):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.shrinkage_value = learning_rate
        self.max_depth = max_depth

        self.loss = loss

        self.ensembles = []
        self.classes = None

    def fit(self, X, y):
        grad = y

        self.classes = np.sort(np.unique(y))

        cart = CART(min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, shrinkage=True)
        self.ensembles.append(cart.fit(X, grad))

        for i in range(self.n_estimators):
            print i,
            aN_1 = np.sum(np.array([b.predict(X) for b in self.ensembles]), axis=0)
            gN = logic_loss(y, aN_1)
            cart = CART(min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, shrinkage=True)
            bN = cart.fit(X, gN)
            self.ensembles.append(bN)

    def predict_proba(self, X, n_estimators=None):
        pred = np.sum(np.array([b.predict(X) for b in self.ensembles[:n_estimators]]), axis=0)
        return np.vstack((neg(pred), pos(pred)))

    def predict(self, X, n_estimators=None):
        proba = self.predict_proba(X, n_estimators)
        return self.classes[np.argmax(proba, axis=0)]
