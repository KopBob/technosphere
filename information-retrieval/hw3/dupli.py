import struct
import sys
import gzip
import cStringIO
import unicodedata
import itertools
from collections import OrderedDict, defaultdict

import mmh3
import html2text

import src.document_pb2 as document_pb2


def parse_html(content):
    h2t = html2text.HTML2Text()
    h2t.ignore_links = True
    h2t.ignore_images = True
    h2t.images_to_alt = False

    return h2t.handle(content)


class DocumentStreamReader:
    def __init__(self, stream):
        self.stream = stream

    def __iter__(self):
        while True:
            sb = self.stream.read(4)
            if sb == '':
                return

            size = struct.unpack('i', sb)[0]
            msg = self.stream.read(size)
            doc = document_pb2.document()
            doc.ParseFromString(msg)
            yield doc


tbl = {i: u' ' for i in xrange(sys.maxunicode)
       if unicodedata.category(unichr(i)).startswith('P') or \
       unicodedata.category(unichr(i)).startswith('N')}

list2comb = lambda l: list(itertools.combinations(l, 2))


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


def doc2counter(documents, ids):
    grouped_docs = defaultdict(list)
    for doc_id, document in documents.iteritems():
        for shingle in document['sketch']:
            grouped_docs[shingle].append(doc_id)
    del documents

    grouped_docs_cleaned = filter(lambda l: len(l) > 1, grouped_docs.itervalues())
    grouped_docs_cleaned = map(lambda x: sorted(x), grouped_docs_cleaned)
    paired_docs = map(list2comb, grouped_docs_cleaned)

    paired_docs_reduced = itertools.chain.from_iterable(paired_docs)

    counter = defaultdict(int)
    for k in paired_docs_reduced:
        counter[k] += 1

    counter_filtered = [(ids[id1], ids[id2], (float(c) / (float(c) + 2 * (20.0 - c)))) for ((id1, id2), c) in
                        counter.iteritems() if
                        c > 14]
    counter_cleaned = sorted(counter_filtered, key=lambda x: x[2], reverse=True)

    return counter_cleaned


def all_in_one(dataset):
    documents_ids = OrderedDict()
    documents = defaultdict(dict)

    for part in dataset[:]:
        with gzip.open(part, 'rb') as f:
            buffer = cStringIO.StringIO(f.read())
            reader = DocumentStreamReader(buffer)

        for i, doc in enumerate(reader):
            doc_id = mmh3.hash(doc.url)

            documents_ids[doc_id] = doc.url

            documents[mmh3.hash(doc.url)] = {
                'url': doc.url,
                'sketch': shingles2sketch(tokens2shingles(text2tokens(doc.text.strip()))),
                #                 'sketch': shingles2sketch(tokens2shingles(text2tokens(parse_html(doc.body).strip()))),
                #                 'tokens': text2tokens(doc.text.strip()),
                #                 'tokens': text2tokens(parse_html(doc.body).strip()),
            }

    result = doc2counter(documents, documents_ids)

    for row in result:
        print "%s %s %f" % row
