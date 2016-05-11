#!/usr/bin/env python
# coding=utf-8

import sys, json

from sklearn.externals import joblib

from feature_extraction import extract_features_from_unlabeled_data

transformer = joblib.load("./models/transformer")
model = joblib.load("./models/model_1")


def splitParagraph(paragraph):
    data_dict = extract_features_from_unlabeled_data(paragraph)

    X = transformer.transform(data_dict)

    y_pred = model.predict(X)

    sentences = []
    prev_pos = 0
    for i, s in enumerate(data_dict):
        if y_pred[i] == -1:
            sentences.append(paragraph[prev_pos:s["_pos"] + 2].strip().encode('utf-8'))
            prev_pos = s["_pos"] + 2

    sentences.append(paragraph[prev_pos:].strip().encode('utf-8'))

    return {'Paragraph': paragraph.encode('utf-8'), 'Sentences': sentences}


def splitParagraph_old(para):
    res = []

    cands = para.split('.')
    r = cands[0]
    for c in range(1, len(cands)):
        if cands[c].startswith(' '):
            res.append(r + '.')
            r = cands[c]
        else:
            r += '.' + cands[c]
    res.append(r)

    return {'Paragraph': para, 'Sentences': res}


for s in sys.stdin:
    print json.dumps(splitParagraph(s.strip().decode('utf-8')), ensure_ascii=False)
