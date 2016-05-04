# coding: utf-8

import sys
import random
import re
import base64

from collections import defaultdict

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.externals import joblib

from HTMLParser import HTMLParser


from constants import STOP_WORDS


def extract_russian(arr):
    if len(arr) == 0: return []
    text = " ".join(arr)
    russian_words = re.findall(u"[\u0400-\u0500]+", text.lower())
    return [w for w in russian_words if w not in STOP_WORDS]

def extract_features(html):
    parser = SpamHTMLParser()
    parser.feed(html)
    bag = []

    for key, val in parser.data.items():
        words = extract_russian(val)
        if len(words) != 0:
            bag += words

    return bag

hash_vect = None
clf = None


def is_spam(pageInb64, url):
    global hash_vect
    global clf

    if hash_vect is None:
        hash_vect = joblib.load('./models/hashing_vectorizer')

    if clf is None:
        clf = joblib.load('./models/model_2_pa')

    html = base64.b64decode(pageInb64).decode('utf-8')
    features = extract_features(html)


    X = hash_vect.transform([' '.join(features)])
    y = clf.predict(X)

    return 0 if y == -1 else 1
