from __future__ import print_function
import sys
import unicodedata
import itertools
from collections import OrderedDict, defaultdict

import mmh3

from src.docreader import DocumentStreamReader

import psutil

curr_mem = lambda: psutil.virtual_memory().used / 1024.0 / 1024.0

tbl = {i: u' ' for i in xrange(sys.maxunicode)
       if unicodedata.category(unichr(i)).startswith('P') or \
       unicodedata.category(unichr(i)).startswith('N')}

list2comb = lambda l: itertools.combinations(l, 2)


def text2tokens(text):
    if len(text) == 0:
        return []
    text_cleaned = text.replace("nbsp_place_holder", " ").translate(tbl)
    tokens = text_cleaned.split()
    return tokens


def tokens2shingles(tokens, step=5):
    n_shingles = len(tokens) + 1 - step
    n_shingles = n_shingles if n_shingles > 0 else 1
    shingles = [" ".join(tokens[i:i + step]) for i in range(n_shingles)]
    return shingles


def shingles2sketch(shingles, m_baskets=20):
    baskets = defaultdict(lambda: -float("inf"))
    for shingle in shingles:
        h = mmh3.hash(shingle.encode('utf8'))
        if baskets[h % m_baskets] < h:
            baskets[h % m_baskets] = h
    return sorted(baskets.values())


def doc2counter(docs_sketches, docs_ids):
    grouped_docs = defaultdict(list)

    for doc_id in docs_sketches.keys():
        for shingle in docs_sketches[doc_id]:
            grouped_docs[shingle].append(doc_id)

        del docs_sketches[doc_id]
    grouped_docs_cleaned = filter(lambda l: len(l) > 1, grouped_docs.itervalues())

    grouped_docs_cleaned = map(lambda x: sorted(x), grouped_docs_cleaned)
    paired_docs = map(list2comb, grouped_docs_cleaned)

    paired_docs_reduced = itertools.chain.from_iterable(paired_docs)

    counter = defaultdict(int)
    for k in paired_docs_reduced:
        counter[k] += 1

    ratio = lambda c: 1.0 * c / (1.0 * c + 2 * (20.0 - c))

    counter_filtered = [(docs_ids[id1], docs_ids[id2], ratio(c)) for ((id1, id2), c) in counter.iteritems() if
                        c > 17]
    counter_cleaned = sorted(counter_filtered, key=lambda x: x[2], reverse=True)

    return counter_cleaned


def all_in_one(dataset):
    docs_ids = OrderedDict()
    docs_sketches = defaultdict(dict)

    for part in dataset[:]:
        reader = DocumentStreamReader(part)
        for i, doc in enumerate(reader):
            doc_id = mmh3.hash(doc.url)

            docs_ids[doc_id] = doc.url

            docs_sketches[mmh3.hash(doc.url)] = shingles2sketch(tokens2shingles(text2tokens(doc.text.strip())))

    result = doc2counter(docs_sketches, docs_ids)

    for row in result:
        print("%s %s %f" % row)
