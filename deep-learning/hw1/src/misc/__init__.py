import numpy as np
from sklearn.preprocessing import LabelBinarizer, normalize


def convert2readable(x, norm=False):
    if norm:
        x = normalize(x.astype(np.float64))

    if len(x.shape) > 1:
        n_samples, d_features = x.shape
        _x = x.reshape((n_samples, d_features, 1)).astype(np.float64)
    else:
        n_samples = x.shape[0]
        _x = x.reshape((n_samples, 1)).astype(np.float64)

    return _x


def transform_data(data):
    x, y = zip(*data)
    y = LabelBinarizer().fit_transform(y)

    x = normalize(np.array(x, dtype=np.float64))
    y = np.array(y, dtype=np.float64)

    return zip(x, y)
