import sys
import codecs
import unicodedata

from nltk.corpus import stopwords


def readfile(filepath, delim=" ", with_none=True):
    with codecs.open(filepath, 'r', "utf-8") as f:
        for line in f:
            data = line.split(delim)
            term = data[0]
            docs = data[1:]
            docs[-1] = docs[-1][:-1]

            yield term, [int(d) for d in docs]
    if with_none:
        while (1): yield None


tbl = {i: u' ' for i in xrange(sys.maxunicode)
       if unicodedata.category(unichr(i)).startswith('P') or \
       unicodedata.category(unichr(i)).startswith('N')}


def text2tokens(text):
    if len(text) == 0:
        return []
    text_cleaned = text.lower().replace("nbsp_place_holder", " ").translate(tbl)
    tokens = text_cleaned.split()
    tokens = [t for t in set(tokens) if t not in stopwords.words('russian')]
    return tokens


def cumdiff(a):
    a = [0] + a
    for i in range(1, len(a)):
        yield a[i] - a[i - 1]
