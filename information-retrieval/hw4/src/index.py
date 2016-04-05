import sys
import codecs

from collections import defaultdict

from .ts_idx.docreader import DocumentStreamReader
from .constants import PATH_TO_DATABASE, PATH_TO_DOC_IDS_FILE, PATH_TO_INDEX_STORAGE
from .misc import text2tokens, readfile

curr_id = 1


def create_inverted_index_from_file(path_to_file):
    global curr_id
    print "Preparing %s ..." % path_to_file

    documents_ids = defaultdict(int)
    inverted_index = defaultdict(list)

    # create index
    reader = DocumentStreamReader(PATH_TO_DATABASE + path_to_file)
    for doc in reader:
        doc_id = curr_id  # hash(doc.url)
        documents_ids[doc_id] = doc.url

        for t in text2tokens(doc.text):
            inverted_index[t].append(doc_id)

        curr_id += 1

    # save docs ids
    with codecs.open(PATH_TO_DOC_IDS_FILE, "a", "utf-8") as f:
        for doc_id, doc_url in documents_ids.iteritems():
            f.write("%s %s\n" % (doc_id, doc_url))

    # save inverted index
    inverted_index_sorted = sorted(inverted_index.iteritems(), key=lambda x: x[0])

    file_hash_name = hash(path_to_file) % ((sys.maxsize + 1) * 2)

    with codecs.open(PATH_TO_INDEX_STORAGE + "%s.txt" % file_hash_name, "w", "utf-8") as f:
        for term, docs in inverted_index_sorted:
            f.write(
                    "%s %s\n" % (
                        term, " ".join(str(x) for x in docs)
                    )
            )


def merge(gen1, gen2):
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



def merge_indexes(pathes):
    print "Intut: ", pathes
    if len(pathes) == 1:
        return pathes[0]

    merged_pathes = []
    for i in range(0, len(pathes), 2):
        part = pathes[i:i + 2]
        if len(part) == 1:
            merged_pathes += part
            continue

        # merge
        print "  Merging: ", part
        part0 = readfile(PATH_TO_INDEX_STORAGE + part[0])
        part1 = readfile(PATH_TO_INDEX_STORAGE + part[1])

        part0_name = part[0].split(".")[0]
        part1_name = part[1].split(".")[0]

        file_hash_name = hash(part0_name + part1_name) % ((sys.maxsize + 1) * 2)
        merged_file_name = "%s" % file_hash_name + ".txt"

        merged_gen = merge(part0, part1)

        with codecs.open(PATH_TO_INDEX_STORAGE + merged_file_name, "w", "utf-8") as f:
            for term, docs in merged_gen:
                f.write(
                        "%s %s\n" % (
                            term, " ".join(str(x) for x in docs)
                        )
                )

        merged_pathes.append(merged_file_name)

    return merge_indexes(merged_pathes)
