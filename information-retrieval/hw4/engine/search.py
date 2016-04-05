#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function

import sys

from src.searcher import Searcher

if __name__ == '__main__':
    data_files = sys.argv[1]

    searcher = Searcher(path_to_inverted_index="../storage/inverted_index.bin",
                        path_to_term_dictionary="../storage/term_dictionary.bin",
                        path_to_documents_registry="../storage/documents_registry.txt")

    for d in data_files.split("\n"):
        searcher.search(d.decode('utf-8'))
