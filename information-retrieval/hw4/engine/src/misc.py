# coding=utf-8
import sys
import codecs
import unicodedata
import struct
import itertools
import collections


# from nltk.corpus import stopwords

def gen_struct_unpack(bytestream, code):
    for b in bytestream:
        unpacked = struct.unpack(code, b)
        if len(unpacked) > 1:
            yield unpacked
        else:
            yield unpacked[0]


def gen_bytes_from_file(filename, seek_offset=0, max_bytes=None, by=1, chunksize=8192):
    if chunksize % by != 0:
        raise ValueError("chunksize % by != 0")

    f = open(filename, "rb")
    f.seek(seek_offset)

    try:
        k_bytes = 0
        while True:
            chunk = f.read(chunksize)

            if chunk:
                for i in range(0, len(chunk), by):
                    yield chunk[i:i + by]  # what if chunksize mod by != 0
                    k_bytes += by

                    if max_bytes is not None and k_bytes >= max_bytes:
                        raise StopIteration
            else:
                raise StopIteration
    except StopIteration:
        pass
    finally:
        f.close()


def readfile(filepath, delim=" ", with_none=True):
    with codecs.open(filepath, 'r', "utf-8") as f:
        for line in f:
            data = line.split(delim)
            term = data[0]
            docs = data[1:]
            docs[-1] = docs[-1][:-1]
            yield term, [int(d) for d in docs]
    if with_none:
        while True:
            yield None


tbl = {i: u' ' for i in xrange(sys.maxunicode)
       if unicodedata.category(unichr(i)).startswith('P') or \
       unicodedata.category(unichr(i)).startswith('N')}


def text2tokens(text):
    if len(text) == 0:
        return []
    text_cleaned = text.lower().translate(tbl)
    tokens = text_cleaned.split()
    # tokens = [t for t in set(tokens) if t not in stopwords.words('russian')]
    return set(tokens)


def cumdiff(a):
    a = [0] + a
    for i in range(1, len(a)):
        yield a[i] - a[i - 1]


def binary_file_reader(path, structure, bytes):
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(bytes), ''):
            yield struct.unpack(structure, chunk)


def str_to_bool(s):
    if s == 'True' or s is True:
        return True
    elif s == 'False' or s is False:
        return False
    else:
        raise ValueError


def consume(iterator, n):
    collections.deque(itertools.islice(iterator, n))
