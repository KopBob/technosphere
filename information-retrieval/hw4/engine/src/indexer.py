from __future__ import print_function

import sys

import os
import struct
import codecs
import time
from collections import defaultdict

import mmh3

from .misc import text2tokens, readfile, cumdiff
from .encoders import VarByte


class Indexer:
    def __init__(self, path_to_storage, flush_threshold=2):
        """
        flush_threshold in Mb
        """
        self.buff_documents_registry = defaultdict(int)
        self.buff_inverted_index = defaultdict(list)

        self.docs_counter = 1
        self.terms_counter = 1

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

        # print "flush_threshold - ", self.flush_threshold

    def store_index(self, index):
        stamp = int(time.time()) + 10
        tmp_inverted_index = self.path_tmp_index_pattern % stamp

        with codecs.open(tmp_inverted_index, "w", "utf-8") as f:
            for term, docs_ids in index:
                f.write(
                        "%s %s\n" % (
                            term, " ".join(str(x) for x in docs_ids)
                        )
                )

        print("   dump ", tmp_inverted_index, file=sys.stderr)

        self.tmp_files.append(tmp_inverted_index)

        return tmp_inverted_index

    def flush(self):
        with codecs.open(self.path_documents_registry, "a", "utf-8") as f:
            for doc_id, doc_url in self.buff_documents_registry.iteritems():
                f.write("%s %s\n" % (doc_id, doc_url))

        buff_terms_registry_sorted = sorted(self.buff_inverted_index.iteritems(), key=lambda x: x[0])

        self.store_index(buff_terms_registry_sorted)

        self.buff_documents_registry.clear()
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
            self.terms_counter += len(doc_tokens)

            # sys.stdout.write('\r' + "{0} {1} {2}".format(self.docs_counter, self.terms_counter, self.flush_indicator))
            # sys.stdout.flush()

            if self.flush_indicator > self.flush_threshold:
                self.flush()

        # self.flush()

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
                # merge
                val = val1

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

        # print "input ", files
        merged_files = []

        for i in range(0, len(files), 2):
            curr_files = files[i:i + 2]

            if len(curr_files) == 1:
                merged_files += curr_files
                continue

            index1 = readfile(curr_files[0])
            index2 = readfile(curr_files[1])

            # print "   merging ", curr_files[0], curr_files[1]

            merged_index = self._merge_two_indexes(index1, index2)

            merged_index_file = self.store_index(merged_index)
            # print "   merged to", merged_index_file
            os.remove(curr_files[0])
            os.remove(curr_files[1])
            # print "   remove", curr_files[0], curr_files[1]

            merged_files.append(merged_index_file)

        return self._merge_all_indesex(merged_files)


class IndexEncoder:
    def __init__(self, path_to_inverted_index, path_to_term_dictionary,
                 encoder=VarByte):
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
                    b_body = bytearray(self.encoder.encode(cumdiff(docs_ids)))

                    term_hash = mmh3.hash(term.encode('utf-8'))
                    b_info = struct.pack('ii', term_hash, len(b_body))

                    # td_f.write("%s;%s;%s\n" % (term, curr_pos, len(bytestream)))
                    td_f.write(b_info)
                    ii_f.write(b_body)

                    # curr_pos += len(b_body)
        os.remove(path_to_raw_index)
