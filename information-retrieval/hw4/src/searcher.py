import mmap
import array

import mmh3
import numpy as np

from src.encoders import VarByte
from src.misc import binary_file_reader
from src.bool_parser import BoolQueryParser


class Searcher:
    def __init__(self, path_to_inverted_index,
                 path_to_term_dictionary,
                 encoder=VarByte,
                 query_parser=BoolQueryParser):
        self.encoder = encoder()
        self.path_to_inverted_index = path_to_inverted_index
        self.path_to_term_dictionary = path_to_term_dictionary
        self.query_parser = query_parser()

        self.terms_meta = {}

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
        words = [w.strip().lower() for w in query.split("&")]
        words_meta = self.get_terms_meta(words)

    def get_term_documents(self, pos, offset, **args):
        with open(self.path_to_inverted_index, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

            raw_data = array.array("B")
            raw_data.fromstring(mm[pos:pos + offset])
            return np.cumsum(list(self.encoder.decode(raw_data)))
