# coding: utf-8

import base64

from sklearn.externals import joblib
from feature_extractor import HTMLDataExtractor, HTMLFeatureExtractor
from sklearn.feature_extraction import DictVectorizer

clf = None


def is_spam(pageInb64, url):
    global clf

    if clf is None:
        clf = joblib.load('./models/gbc_model')

    html = base64.b64decode(pageInb64).decode('utf-8')

    html_de = HTMLDataExtractor()
    html_data = html_de.extract(html)

    html_fe = HTMLFeatureExtractor()
    html_fe.extract(html_data)

    v = DictVectorizer(sparse=False)
    x = v.fit_transform([html_fe.features])

    y = clf.predict(x)

    return y
