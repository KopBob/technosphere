#!/usr/bin/env python

# coding=utf-8


from __future__ import print_function

import time

import sys

from src.indexer import Indexer
from src.indexer import IndexEncoder

from src.ts_idx.docreader import DocumentStreamReader

if __name__ == '__main__':
    encoding_type = sys.argv[1]
    data_files = sys.argv[2:]

    indexer = Indexer("../storage/", flush_threshold=50)

    start = time.time()

    for data_file in data_files:
        documents_stream = DocumentStreamReader(data_file)
        indexer.index(documents_stream)

    end = time.time()
    print("Indexing ", end - start, file=sys.stderr)

    start = time.time()
    path_to_merged_index = indexer.merge()
    end = time.time()
    print("Merge ", end - start, file=sys.stderr)

    start = time.time()
    index_encoder = IndexEncoder("../storage/inverted_index.bin", "../storage/term_dictionary.bin")
    index_encoder.encode(path_to_merged_index)
    end = time.time()
    print("Encode", end - start, file=sys.stderr)
