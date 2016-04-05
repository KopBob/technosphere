import mmap
import array

import mmh3
import numpy as np

from src.encoders import VarByte
from src.misc import binary_file_reader
from src.bool_parser import BoolQueryParser


def load_documents_registry(path_to_dr):
    documents_registry = {}

    with open(path_to_dr, 'r') as f:
        for line in f.readlines():
            data = line.split(" ")
            doc_id = int(data[0])
            doc_url = data[1][:-1]
            documents_registry[doc_id] = doc_url

    return documents_registry


class Searcher:
    def __init__(self, path_to_inverted_index,
                 path_to_term_dictionary,
                 path_to_documents_registry,
                 encoder=VarByte,
                 query_parser=BoolQueryParser):
        self.encoder = encoder()
        self.path_to_inverted_index = path_to_inverted_index
        self.path_to_term_dictionary = path_to_term_dictionary
        self.query_parser = query_parser(get_term_data=self.get_documents)
        self.documents_registry = load_documents_registry(path_to_documents_registry)

        self.terms_meta = {}

    def get_documents(self, term):
        term_meta = self.get_term_meta(term)
        term_docs = self.get_term_documents(**term_meta)
        return term_docs

    def get_term_meta(self, term):
        term_hash = mmh3.hash(term.encode("utf-8"))

        ans = {}
        curr_pos = 0
        dictionary_reader = binary_file_reader(self.path_to_term_dictionary, "ii", 8)
        for dict_term_hash, size in dictionary_reader:

            if dict_term_hash == term_hash:
                ans["pos"] = curr_pos
                ans["offset"] = size
                break

            curr_pos += size

        return ans

    def get_terms_meta(self, query_token):
        query_tokens_dict = {mmh3.hash(token.encode("utf-8")): {"token": token} for token in query_token}

        dictionary_reader = binary_file_reader(self.path_to_term_dictionary, "ii", 8)

        curr_pos = 0
        unprocessed_query_hashes = query_tokens_dict.keys()
        for term_hash, size in dictionary_reader:

            if term_hash in unprocessed_query_hashes:
                query_tokens_dict[term_hash]["pos"] = curr_pos
                query_tokens_dict[term_hash]["offset"] = size

                unprocessed_query_hashes.remove(term_hash)

            if not len(unprocessed_query_hashes):
                break

            curr_pos += size

        return query_tokens_dict

    def search(self, query):
        results = self.query_parser.parse_query(query)

        print query
        print len(results)
        for res in results:
            print self.documents_registry[res]

    def get_term_documents(self, pos=None, offset=None, **args):
        if pos is None:
            return set([])
        with open(self.path_to_inverted_index, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

            raw_data = array.array("B")
            raw_data.fromstring(mm[pos:pos + offset])
            return set(np.cumsum(list(self.encoder.decode(raw_data))))
