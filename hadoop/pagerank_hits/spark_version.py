#!/usr/bin/env python

# coding: utf-8

#------------------------------------------------------------------------------------------

sc

#------------------------------------------------------------------------------------------

# from __future__ import print_function

import re
import sys
from operator import add

#------------------------------------------------------------------------------------------

base_path = "./1_10/"
# base_path = "./lenta.ru/1_1000/"

# # 1. Parse documents

#------------------------------------------------------------------------------------------

import pandas as pd
urls_df = pd.read_csv(base_path + 'urls.txt', sep=str('\t'), header=None)
urls_df.columns = ["doc_id", "url"]
urls_df = urls_df.set_index("doc_id")
doc_urls_dict = urls_df.to_dict()["url"]

#------------------------------------------------------------------------------------------

from base64 import b64decode
import zlib
from bs4 import BeautifulSoup

doc_urls_dict_bc = sc.broadcast(doc_urls_dict)


def parse_doc_line(line):
    doc_id, doc_base64 = line.split('\t')
    doc_gzip = b64decode(doc_base64)
    doc_html = zlib.decompress(doc_gzip).decode('utf-8')

    return int(doc_id), doc_html


def extract_links(doc):
    doc_id, doc_html = doc

    doc_links = []
    boc_bs = BeautifulSoup(doc_html, 'lxml')
    for a in boc_bs.find_all('a', href=True):
        href = a['href']
        if href.startswith('mailto'):
            continue
        if href.startswith('/'):
            href = 'http://lenta.ru' + href
        if not href.startswith('http') and not href.startswith('www'):
            pass
        else:
            doc_links.append(href.strip())

    return doc_urls_dict_bc.value[doc_id].strip(), doc_links


lines = sc.textFile(base_path + "docs*")
docs = lines.map(lambda line: parse_doc_line(line))
doc_links = docs.map(lambda doc: extract_links(doc))
doc_links = doc_links.filter(lambda x: len(x[1]))
adjacency_list = doc_links.flatMapValues(lambda x: x)
to_save = adjacency_list.map(lambda urls: "%s\t%s" % urls)
to_save = to_save.distinct()
out_path_stage1 = './out_stage1/%s' % base_path.replace('/', '_').replace('.',
                                                                          '_')
to_save.saveAsTextFile(out_path_stage1)

# # 2. Page Rank

#------------------------------------------------------------------------------------------

N = sc.textFile(base_path + "docs*").count()
gamma = 0.85

#------------------------------------------------------------------------------------------

N

#------------------------------------------------------------------------------------------


def compute_contribs(urls, rank):
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


def parse_neighbors(urls):
    parts = urls.split('\t')
    return parts[0], parts[1]


#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

lines = sc.textFile(out_path_stage1 + '/part-*')
doc_links = lines.map(lambda urls: parse_neighbors(urls)).distinct(
).groupByKey().cache()

#------------------------------------------------------------------------------------------

ranks = doc_links.map(lambda url_neighbors: (url_neighbors[0], 1 / float(N)))

#------------------------------------------------------------------------------------------

for iteration in range(4):
    contribs = doc_links.join(ranks).flatMap(
        lambda url_urls_rank: compute_contribs(url_urls_rank[1][0], url_urls_rank[1][1])
    )

    ranks = contribs.reduceByKey(add).mapValues(
        lambda sum_contrib: sum_contrib * gamma + (1 - gamma) / float(N))

#------------------------------------------------------------------------------------------

ranks_sorted = ranks.sortBy(lambda a: a[1], ascending=False)

#------------------------------------------------------------------------------------------

# out_path_stage2 = './out_stage2/%s' % base_path.replace('/', '_').replace('.', '_')
# to_save = ranks_sorted.map(lambda x: "%s\t%s" % x)
# to_save.saveAsTextFile(out_path_stage2)

#------------------------------------------------------------------------------------------

for (link, rank) in ranks_sorted.take(30):
    print("%s has rank: %s." % (link, rank))

# # 3 HIST

#------------------------------------------------------------------------------------------

import math

#------------------------------------------------------------------------------------------

lines = sc.textFile(out_path_stage1 + '/part-*')
doc_links = lines.map(lambda urls: parse_neighbors(urls)).distinct(
).groupByKey().cache()
lines = sc.textFile(out_path_stage1 + '/part-*')
invert_doc_links = lines.map(
    lambda urls: parse_neighbors(urls)[::-1]).distinct().groupByKey().cache()

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

auth_score = invert_doc_links.map(
    lambda url_neighbors: (url_neighbors[0], 1 / math.sqrt(N)))
hub_score = doc_links.map(
    lambda url_neighbors: (url_neighbors[0], 1 / math.sqrt(N)))

#------------------------------------------------------------------------------------------

for _ in range(1):
    # update all authority values first
    hub_contribs = invert_doc_links.join(auth_score).flatMap(
        lambda url_urls_rank: compute_contribs(url_urls_rank[1][0], url_urls_rank[1][1])
    )

    hub_score = hub_contribs.reduceByKey(add).mapValues(lambda x: x)

    hub_norm = math.sqrt(hub_score.map(lambda x: x[1]**2).sum())
    hub_norm_bc = sc.broadcast(hub_norm)
    hub_score = hub_score.map(lambda x: (x[0], x[1] / hub_norm_bc.value))

    # then update all hub values
    auth_contribs = doc_links.join(hub_score).flatMap(
        lambda url_urls_rank: compute_contribs(url_urls_rank[1][0], url_urls_rank[1][1])
    )

    auth_score = auth_contribs.reduceByKey(add).mapValues(lambda x: x)

    auth_norm = math.sqrt(auth_score.map(lambda x: x[1]**2).sum())
    auth_norm_bc = sc.broadcast(auth_norm)
    auth_score = auth_score.map(lambda x: (x[0], x[1] / auth_norm_bc.value))

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

auth_score_sorted = auth_score.sortBy(lambda a: a[1], ascending=False)
hub_score_sorted = hub_score.sortBy(lambda a: a[1], ascending=False)

#------------------------------------------------------------------------------------------

for (link, rank) in hub_score_sorted.take(30):
    print("%s has hub: %s." % (link, rank))

#------------------------------------------------------------------------------------------

for (link, rank) in auth_score_sorted.take(30):
    print("%s has authority: %s." % (link, rank))

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
