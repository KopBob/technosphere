#!/usr/bin/env python

# coding: utf-8

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------


class Document(object):
    def __init__(self, url, title_text, body_text):
        self.url = url
        self.title_text = title_text
        self.body_text = body_text


#------------------------------------------------------------------------------------------

import os
# Get pure files paths
files_paths = []
for root, dirs, files in os.walk("../data/"):
    path = root.split('/')

    for f in files:
        if "DS_Store" not in f:
            path_to_file = path + [f]
            files_paths.append('/'.join(path_to_file))

# get url form files
files_urls = []
for f in files_paths:
    with open(f, 'r') as fin:
        first_line = fin.readline().strip()
        files_urls.append(first_line)

import pandas as pd

# creade document registry
doc_registry_dt = pd.DataFrame(
    zip(files_paths, files_urls,
        map(lambda x: '_'.join(x.split('/')[-2:]), files_paths)),
    columns=["pure_path", "url", "id"])

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("../parsed/") if isfile(join("../parsed/", f))]
parsed = [
    '../data/' +
    '/'.join(f.replace('html-', '').replace('txt-', '').split('_'))
    for f in onlyfiles
]
doc_registry_dt.loc[doc_registry_dt.pure_path.isin(parsed), "is_parsed"] = True

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

import zlib

#------------------------------------------------------------------------------------------

import sys

#------------------------------------------------------------------------------------------


def gen_docs():
    for i, (pure_path, url, doc_id, is_parsed) in doc_registry_dt.iterrows():
        if is_parsed != True:
            continue
        sys.stdout.write('\r ' + "%s / %s" % (i, doc_registry_dt.shape[0]))
        with open("../parsed/" + doc_id, 'r') as fin:
            file_content = zlib.decompress(fin.read())
            text_parts = file_content.split("!@#$%^&*()")
            title_text = text_parts[0]
            body_text = text_parts[1]

        doc = Document(url,
                       unicode(title_text.decode('utf-8')),
                       unicode(body_text.decode('utf-8')))
        yield doc


#------------------------------------------------------------------------------------------

from engine.src import indexer

#------------------------------------------------------------------------------------------

dexter = indexer.Indexer("../storage/", flush_threshold=10000)

#------------------------------------------------------------------------------------------

dexter.index(gen_docs())

#------------------------------------------------------------------------------------------

import numpy as np

#------------------------------------------------------------------------------------------

print("hello")

#------------------------------------------------------------------------------------------

from __future__ import print_function

#------------------------------------------------------------------------------------------

from collections import defaultdict

#------------------------------------------------------------------------------------------

avgdl = np.mean([l for _, (_, l) in dexter.buff_documents_registry.items()])

#------------------------------------------------------------------------------------------

k1 = 1.2
b = 0.75

#------------------------------------------------------------------------------------------

doc_score = defaultdict(float)
for w in [u"путин", u"о", u"чурках"]:
    print(w)
    D = len(dexter.buff_documents_registry.keys())
    df_t_D = len(dexter.buff_inverted_index[w])
    idf = np.log(D / (1 + float(df_t_D)))

    for doc, tf, is_in_title in dexter.buff_inverted_index[w]:
        dl = dexter.buff_documents_registry[doc][1]
        #         score = idf * ((tf*(10 + is_in_title) * (k1 + 1)) / (tf*(10 + is_in_title) + k1*(1 - b + b*(dl/avgdl))))
        doc_score[doc] += (1 + np.log(tf)) * idf
#         print(doc, (1 + np.log(tf))*idf)
sorted(doc_score.items(), key=lambda x: x[1], reverse=True)[:10]

#------------------------------------------------------------------------------------------

sorted(doc_score.items(), key=lambda x: x[1], reverse=True)

#------------------------------------------------------------------------------------------

avgdl = np.mean([l for _, (_, l) in dexter.buff_documents_registry.items()])
k1 = 1.2
b = 0.75


def score_q(query):
    doc_score = defaultdict(float)
    for w in query.lower().split():
        D = len(dexter.buff_documents_registry.keys())
        df_t_D = len(dexter.buff_inverted_index[w])
        idf = np.log(D / (1 + float(df_t_D)))

        for doc, tf, is_in_title in dexter.buff_inverted_index[w]:
            dl = dexter.buff_documents_registry[doc][1]
            score = idf * ((tf * (1 + is_in_title) * (k1 + 1)) /
                           (tf * (10 + is_in_title) + k1 * (1 - b + b *
                                                            (dl / avgdl))))
            doc_score[doc] += (1 + np.log(tf)) * idf
#             print(doc, (1 + np.log(tf))*idf)
    return sorted(doc_score.items(), key=lambda x: x[1], reverse=True)


#------------------------------------------------------------------------------------------

queries_df = pd.read_csv(
    '../../hw3/queries.numerate.txt',
    sep=str('\t'),
    encoding="utf-8",
    header=None,
    names=["num", 'query'],
    index_col="num")
urls_df = pd.read_csv(
    '../../hw3/urls.numerate.txt',
    sep=str('\t'),
    encoding="utf-8",
    header=None,
    names=["num", 'url'],
    index_col="num")
dexter_doc_df = pd.DataFrame(
    [(doc_id, url)
     for doc_id, (url, _) in dexter.buff_documents_registry.items()],
    columns=["doc_id", "url"])
dexter_doc_df = dexter_doc_df.set_index("doc_id")
docs_df = dexter_doc_df.merge(
    urls_df.reset_index(), how="outer", left_on="url", right_on="url")

#------------------------------------------------------------------------------------------

queries_df.shape

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

subm = pd.read_csv('../../hw3/sample.csv')
subm_g = subm.groupby("QueryId")

#------------------------------------------------------------------------------------------

res = score_q(queries_df.loc[2].query)

#------------------------------------------------------------------------------------------

tt = subm.groupby("QueryId").get_group(2).DocumentId.tolist()

#------------------------------------------------------------------------------------------

[r[0] for r in res if r[0] in tt][:10]

#------------------------------------------------------------------------------------------

import sys

#------------------------------------------------------------------------------------------

submit = []
for i, (q_num, row) in enumerate(queries_df.iterrows()):
    sys.stdout.write("\r  " + "%s | %s" % (q_num, queries_df.shape[0]))
    target_docs = subm_g.get_group(q_num).DocumentId.tolist()
    q = row.query
    res = score_q(q)
    doc_nums = [
        int(docs_df.loc[doc_id].num) for doc_id, score in res
        if not pd.isnull(docs_df.loc[doc_id].num)
    ]
    for doc_num in [d for d in doc_nums if d in target_docs]:
        #     for doc_num in [int(docs_df.loc[doc_id].num) for doc_id, score in res[:10] if not pd.isnull(docs_df.loc[doc_id].num)]:
        submit.append((q_num, doc_num))

#------------------------------------------------------------------------------------------

submit_df = pd.DataFrame(submit, columns=["QueryId", "DocumentId"])

#------------------------------------------------------------------------------------------

submit_df.to_csv("../submission3.csv", index=False)

#------------------------------------------------------------------------------------------

import io

#------------------------------------------------------------------------------------------

import json
with io.open('buff_documents_registry.json', 'wb') as fp:
    json.dump(dexter.buff_documents_registry, fp, indent=4)

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

import json
with io.open('buff_inverted_index.json', 'wb') as fp:
    json.dump(dexter.buff_inverted_index, fp, indent=2)

#------------------------------------------------------------------------------------------

res = score_q(queries_df.loc[2].query)
[int(docs_df.loc[doc_id].num) for doc_id, score in res[:10]]

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

docs_df = dexter_doc_df.merge(
    urls_df.reset_index(), how="outer", left_on="url", right_on="url")

#------------------------------------------------------------------------------------------

docs_df.loc[res[0][0]]

#------------------------------------------------------------------------------------------

urls_df

#------------------------------------------------------------------------------------------
