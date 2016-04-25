# coding=utf-8
import mmap
import array
import re

import mmh3
import numpy as np

from .encoders import VarByte, Simple9
from .misc import binary_file_reader, gen_bytes_from_file, gen_struct_unpack
from .generators import gen_cumsum, gen_empty
from .bool_parser import BoolQueryParser


def select_terms_meta(query_terms, term_dict_stream):
    """
    reads term dictionary generator and selects query terms meta info
    """
    terms_meta_dict = {}

    for term in query_terms:
        term_hash = mmh3.hash(term.encode("utf-8"))
        terms_meta_dict[term_hash] = {
            "term": term,
            "seek_offset": None,
            "size": None
        }

    seek_offset = 0
    unseen_terms = terms_meta_dict.keys()
    for dict_term_hash, dict_term_size in term_dict_stream:
        if dict_term_hash in unseen_terms:
            terms_meta_dict[dict_term_hash]["seek_offset"] = seek_offset
            terms_meta_dict[dict_term_hash]["size"] = dict_term_size

            unseen_terms.remove(dict_term_hash)
            if len(unseen_terms) == 0:
                break

        seek_offset += dict_term_size

    query_terms_dict = {}
    for _, term_meta in terms_meta_dict.items():
        query_terms_dict[term_meta["term"]] = {
            "seek_offset": term_meta["seek_offset"],
            "size": term_meta["size"]
        }

    return query_terms_dict


def gen_find_terms_in_terms_dict(query_terms, path_to_terms_dict):
    """
    input: list of query terms [u"путин", u"медведев"]
    outpu: dict of query terms
    u"путин": {
        "seek_offset": N,
        "size": M
    }, ...
    """
    bytestream = gen_bytes_from_file(path_to_terms_dict, by=8)
    term_dict_stream = gen_struct_unpack(bytestream, code='ii')

    query_terms_dict = select_terms_meta(query_terms, term_dict_stream)

    return query_terms_dict


def gen_term_documents(path, term, decoder):
    if term.get("size") is None:
        return gen_empty()

    bytestream = gen_bytes_from_file(path, term.get("seek_offset"), term.get("size"))
    # numstream = gen_struct_unpack(bytestream, code='B')
    decoded_bytes = decoder.gen_decode(bytestream)
    doc_ids = gen_cumsum(decoded_bytes)

    return doc_ids


def load_documents_registry(path_to_dr):
    documents_registry = {}

    with open(path_to_dr, 'r') as f:
        for line in f.readlines():
            data = line.split(" ")
            doc_id = int(data[0])
            doc_url = data[1][:-1]
            documents_registry[doc_id] = doc_url

    return documents_registry


rx = re.compile('([)(&!|])')


class Searcher2:
    def __init__(self, path_to_inverted_index,
                 path_to_term_dictionary,
                 path_to_documents_registry,
                 encoder=Simple9,
                 query_parser=BoolQueryParser):
        self.encoder = encoder()
        self.path_to_inverted_index = path_to_inverted_index
        self.path_to_term_dictionary = path_to_term_dictionary
        self.query_parser = query_parser()
        self.documents_registry = load_documents_registry(path_to_documents_registry)

        self.terms_meta = {}

    def collect_terms_generators(self, query):
        terms = rx.sub(r'', query).lower().split()
        terms_meta = gen_find_terms_in_terms_dict(terms, self.path_to_term_dictionary)
        terms_gens = {}
        for term, meta in terms_meta.items():
            terms_gens[term] = gen_term_documents(self.path_to_inverted_index, meta, self.encoder)

        return terms_gens

    def evaluate_query(self, query, terms_meta):
        return self.query_parser.parse_query()

    def search(self, query):
        terms_gens = self.collect_terms_generators(query)
        results = self.query_parser.parse_query(query, terms_gens)

        results = list(results)
        print query.encode("utf-8")
        print len(results)
        for res in sorted(results):
            print self.documents_registry[res]




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

        print query.encode("utf-8")
        print len(results)
        for res in sorted(results):
            print self.documents_registry[res]

    def get_term_documents(self, pos=None, offset=None, **args):
        if pos is None:
            return set([])
        with open(self.path_to_inverted_index, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)

            raw_data = array.array("B")
            raw_data.fromstring(mm[pos:pos + offset])

            return set(np.cumsum(list(self.encoder.decode(mm[pos:pos + offset]))))

            #
            # def get_gen_term_documents(self, pos=None, offset=None, **args):
            #     if pos is None:
            #         yield
            #
            #
            #     with open(self.path_to_inverted_index, 'rb') as f:
            #         mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            #
            #         raw_data = array.array("B")
            #         raw_data.fromstring(mm[pos:pos + offset])
            #
            #         return set(np.cumsum(list(self.encoder.decode(raw_data))))
