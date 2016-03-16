# coding: utf-8
import time
import numpy as np

from sklearn.cluster import Birch
from sklearn.feature_selection import SelectKBest

from sklearn.feature_extraction import DictVectorizer

import re
import os

import urlparse
import urllib

from collections import defaultdict, Counter

threshold_filter = lambda counter, threshold: filter(lambda f: f[1] > threshold, counter.most_common())
len_filter = lambda array: filter(len, array)
get_ext = lambda x: os.path.splitext(x)[1][1:]
num_inside = lambda x: True if re.match(r"\D+\d+\D+", x) else False
unquote = lambda segments: map(urllib.unquote_plus, segments)


def lod2dol(lod):
    """list_of_dicts_to_dict_of_lists"""
    result = defaultdict(list)
    for d in lod:
        for k, v in d.items():
            result[k].append(v)
    return result


def segment_ext_substr_num_index(urls, threshold=100):
    segs = [dict(enumerate(unquote(len_filter(url.path.split('/'))))) for url in urls]
    seg_by_index = lod2dol(segs)
    func = lambda x: (num_inside(x), get_ext(x))
    ext_len_filter = lambda x: len(x[1])
    t = {k: Counter(filter(ext_len_filter, map(func, seg_by_index[k]))) for k in seg_by_index.keys()}
    result = []
    for seg, counter in t.items():
        for (flag, ext), count in counter.items():
            if flag:
                if count > threshold:
                    result.append(("segment_ext_substr[0-9]_%d:%s" % (seg, ext), count))
    return result


def segment_substr_num_index(urls, threshold=100):
    segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]
    seg_by_index = lod2dol(segs)
    func = lambda c: c[True] if c[True] > c[False] else 0
    t = {k: Counter(map(num_inside, unquote(seg_by_index[k])))[True] for k in seg_by_index.keys()}
    substr_num_segs = {k: v for k, v in t.items() if v > threshold}
    result = [("segment_substr[0-9]_%d:1" % (seg), count) for seg, count in substr_num_segs.items()]
    return result


def segment_ext_index(urls, threshold=100):
    segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]
    seg_by_index = lod2dol(segs)
    t = {k: threshold_filter(Counter(len_filter(map(get_ext, seg_by_index[k]))), threshold) for k in
         seg_by_index.keys()}
    exts = {k: v for k, v in t.items() if len(v)}
    result = []
    for seg, ext_array in exts.items():
        for ext, count in ext_array:
            result.append(("segment_ext_%s:%s" % (seg, ext), count))
    return result


def extract_segment_is_number(urls, threshold=100):
    segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]
    seg_by_index = lod2dol(segs)
    isnum = lambda x: unicode(x).isnumeric()
    t = {k: Counter(map(isnum, seg_by_index[k]))[True] for k in seg_by_index.keys()}
    numeric_segs = {k: v for k, v in t.items() if v > threshold}
    result = [("segment_[0-9]_%d:1" % seg, count) for seg, count in numeric_segs.items()]
    return result


def extract_segment_len_index(urls, threshold=100):
    segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]
    seg_by_index = lod2dol(segs)
    t = {k: threshold_filter(Counter(map(len, seg_by_index[k])), threshold) for k in seg_by_index.keys()}
    result = []
    for seg, lengths in t.items():
        for length, count in lengths:
            result.append(("segment_len_%d:%d" % (seg, length), count))
    return result


def extract_segment_name_index(urls, threshold=100):
    segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]
    seg_pos = lod2dol(segs)
    t = {k: threshold_filter(Counter(seg_pos[k]), threshold) for k in seg_pos.keys()}
    result = []
    for seg, names in t.items():
        for name, count in names:
            result.append(("segment_name_%d:%s" % (seg, name), count))
    return result


def extract_param(urls, threshold=100):
    queries = sum([len_filter(url.query.split('&')) for url in urls], [])
    queries_counter = Counter(queries)
    queries_cleaned = threshold_filter(queries_counter, threshold)
    result = [("param:%s" % q, c) for q, c in queries_cleaned]
    return result


def extract_param_name(urls, threshold=100):
    param_names = sum([urlparse.parse_qs(url.query).keys() for url in urls], [])
    param_names_counter = Counter(param_names)
    params_cleaned = threshold_filter(param_names_counter, threshold)
    result = [("param_name:%s" % p, c) for p, c in params_cleaned]
    return result


def extract_segments_len(urls, threshold=100):
    count_sg = lambda url: len(len_filter(url.path.split('/')))
    sg_counter = Counter([count_sg(url) for url in urls])
    sg_counts_cleaned = threshold_filter(sg_counter, threshold)
    result = [("segments:%d" % sg, count) for sg, count in sg_counts_cleaned]
    return result


def extract_features(links, threshold=100):
    result = []

    result += extract_segments_len(links, threshold)
    result += extract_param_name(links, threshold)
    result += extract_param(links, threshold)
    result += extract_segment_name_index(links, threshold)
    result += extract_segment_len_index(links, threshold)
    result += extract_segment_is_number(links, threshold)
    # result += segment_ext_index(links, threshold)
    # result += segment_substr_num_index(links, threshold)
    # result += segment_ext_substr_num_index(links, threshold)

    result = sorted(result, key=lambda tup: tup[1], reverse=True)

    return result


parse_url = lambda url: urlparse.urlparse(url.rstrip())

algos = {}

BIRCH_BRANCHING_FACTOR = 30
BIRCH_THRESHOLD = 0.25
KBEST_K = 40


def define_segments(QLINK_URLS, UNKNOWN_URLS, QUOTA):
    # url to obj
    qlinks = map(parse_url, QLINK_URLS)
    ulinks = map(parse_url, UNKNOWN_URLS)

    # check netloc
    # print qlinks[0].netloc

    # extract features
    start = time.time()
    qlinks_f = [dict(Counter(zip(*extract_features([link], 0))[0])) for link in qlinks]
    ulinks_f = [dict(Counter(zip(*extract_features([link], 0))[0])) for link in ulinks]
    # print time.time() - start
    # start = time.time()

    v = DictVectorizer(sparse=False)
    x_ = v.fit_transform(qlinks_f + ulinks_f)

    best_features = np.sum(x_, axis=0) > 5

    m_features = np.sum(best_features)

    v = v.restrict(best_features)
    x_ = x_[:, best_features]

    clustering = Birch(branching_factor=BIRCH_BRANCHING_FACTOR, n_clusters=m_features,
                       threshold=BIRCH_THRESHOLD, compute_labels=True)
    y_ = clustering.fit_predict(x_)

    sel = SelectKBest(k=min(m_features, KBEST_K))
    x = sel.fit_transform(x_, y_)

    y = clustering.fit_predict(x)
    q_or_u = np.repeat([1, 0], [len(QLINK_URLS), len(UNKNOWN_URLS)])
    q_ = np.vstack((y, q_or_u)).T

    quota = zip(np.unique(y),
                (np.array([np.sum(q_[q_[:, 0] == c, 1]) for c in np.unique(y)]) / float(len(QLINK_URLS))) * QUOTA * 2)
    quota = {c: int(q) for c, q in quota}

    algos[qlinks[0].netloc] = {
        "clustering": clustering,
        "quota": quota,
        "sel": sel,
        "vect": v,
        "total_quota": QUOTA,
    }
    # print time.time() - start


def url2vect(link, sel, vect):
    features = extract_features([link], 0)
    features_dict = dict(Counter(zip(*features)[0]))
    return sel.transform(vect.transform(features_dict))


def fetch_url(url):
    link = parse_url(url)
    algo = algos[link.netloc]

    clustering = algo["clustering"]
    quota = algo["quota"]
    total_quota = algo["total_quota"]
    vect = algo["vect"]
    sel = algo["sel"]

    if total_quota <= 0: return False

    link_data = url2vect(link, sel, vect)
    c = clustering.predict(link_data)[0]

    if quota[c] > 0:
        quota[c] -= 1
        total_quota -= 1
        return True

    return False

