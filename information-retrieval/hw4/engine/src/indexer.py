# coding=utf-8

from __future__ import print_function

import sys
import uuid
import os
import struct
import codecs
import time
from collections import defaultdict

import mmh3

from .misc import text2tokens, readfile, cumdiff
from .encoders import VarByte, Simple9


class Indexer:
    def __init__(self, path_to_storage, flush_threshold=2):
        """
        flush_threshold in Mb
        """
        self.buff_documents_registry = defaultdict(int)
        self.buff_inverted_index = defaultdict(list)

        self.docs_counter = 1

        self.flush_indicator = 0
        self.flush_threshold = flush_threshold * 1024 * 1024 / 8.0

        self.path_storage = path_to_storage
        self.path_documents_registry = os.path.join(self.path_storage, "documents_registry.txt")
        self.path_inverted_index = os.path.join(self.path_storage, "inverted_index.bin")
        self.path_term_dictionary = os.path.join(self.path_storage, "term_dictionary.bin")
        self.path_tmp_storage = os.path.join(self.path_storage, "tmp")
        self.path_tmp_index_pattern = os.path.join(self.path_tmp_storage, "%s.txt")

        self.tmp_files = []

        self.path_to_merged_index = None

        try:
            os.makedirs(self.path_storage)
            os.makedirs(self.path_tmp_storage)

            open(self.path_documents_registry, 'w').close()
        except OSError, e:
            if e.errno != 17:
                raise

    def store_index(self, index):
        stamp = uuid.uuid4()

        tmp_inverted_index = self.path_tmp_index_pattern % stamp
        with codecs.open(tmp_inverted_index, "w", "utf-8") as f:
            dump = []
            for term, docs_ids in index:
                dump.append("%s %s" % (term, " ".join(str(x) for x in docs_ids)))
            f.write("\n".join(dump) + "\n")

        self.tmp_files.append(tmp_inverted_index)

        return tmp_inverted_index

    def flush(self):
        if len(self.buff_documents_registry.items()) == 0:
            return

        # flush documents registry
        with codecs.open(self.path_documents_registry, "a", "utf-8") as f:
            dump = []
            for doc_id, doc_url in self.buff_documents_registry.items():
                dump.append("%s %s" % (doc_id, doc_url))
            f.write("\n".join(dump) + "\n")
        self.buff_documents_registry.clear()

        # flush inverted index buffer
        self.store_index(sorted(self.buff_inverted_index.iteritems(),
                                key=lambda x: x[0]))
        self.buff_inverted_index.clear()

        self.flush_indicator = 0

    def index(self, documents_stream):
        for doc in documents_stream:
            doc_id = self.docs_counter

            self.buff_documents_registry[doc_id] = doc.url

            doc_tokens = text2tokens(doc.text)
            for t in doc_tokens:
                self.buff_inverted_index[t].append(doc_id)

            self.docs_counter += 1
            self.flush_indicator += len(doc_tokens) + 1

            if self.flush_indicator > self.flush_threshold:
                self.flush()

    def merge(self):
        self.flush()

        self.path_to_merged_index = self._merge_all_indesex(self.tmp_files[:])
        return self.path_to_merged_index

    def _merge_two_indexes(self, gen1, gen2):
        val1 = gen1.next()
        val2 = gen2.next()

        while (1):
            if val1[0] < val2[0]:
                yield val1
                val1 = gen1.next()

            elif val1[0] == val2[0]:
                yield (val1[0], sorted(val1[1] + val2[1]))
                val1 = gen1.next()
                val2 = gen2.next()

            elif val1[0] > val2[0]:
                yield val2
                val2 = gen2.next()

            if val1 is None:
                while (val2):
                    yield val2
                    val2 = gen2.next()
                break

            if val2 is None:
                while (val1):
                    yield val1
                    val1 = gen1.next()
                break

    def _merge_all_indesex(self, files):
        if len(files) == 1:
            return files[0]

        merged_files = []

        for i in range(0, len(files), 2):
            curr_files = files[i:i + 2]

            if len(curr_files) == 1:
                merged_files += curr_files
                continue

            index1 = readfile(curr_files[0])
            index2 = readfile(curr_files[1])

            merged_index = self._merge_two_indexes(index1, index2)

            merged_index_file = self.store_index(merged_index)
            os.remove(curr_files[0])
            os.remove(curr_files[1])
            merged_files.append(merged_index_file)

        return self._merge_all_indesex(merged_files)


class IndexEncoder:
    def __init__(self, path_to_inverted_index, path_to_term_dictionary,
                 encoder=Simple9):
        self.encoder = encoder()

        self.path_to_inverted_index = path_to_inverted_index
        self.path_to_term_dictionary = path_to_term_dictionary

    def encode(self, path_to_raw_index):
        raw_index = readfile(path_to_raw_index, with_none=False)

        with open(self.path_to_inverted_index, 'wb') as ii_f:
            with open(self.path_to_term_dictionary, "wb") as td_f:
                # prev_pos = 0
                # curr_pos = 0

                for i, (term, docs_ids) in enumerate(raw_index):
                    # b_body = bytearray(self.encoder.encode(cumdiff(docs_ids)))
                    b_body = "".join(self.encoder.encode(cumdiff(docs_ids)))

                    term_hash = mmh3.hash(term.encode('utf-8'))
                    b_info = struct.pack('ii', term_hash, len(b_body))

                    # td_f.write("%s;%s;%s\n" % (term, curr_pos, len(bytestream)))
                    td_f.write(b_info)
                    ii_f.write(b_body)

                    # curr_pos += len(b_body)
        os.remove(path_to_raw_index)
