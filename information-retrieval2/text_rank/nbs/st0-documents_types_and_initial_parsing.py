#!/usr/bin/env python

# coding: utf-8

#------------------------------------------------------------------------------------------

from __future__ import print_function
import os

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

from pyth.plugins.rtf15.reader import Rtf15Reader
from pyth.plugins.plaintext.writer import PlaintextWriter


def rtf2text(rtf_content):
    fp = io.BytesIO(rtf_content)
    doc = Rtf15Reader.read(fp)
    fp.close()
    return PlaintextWriter.write(doc).getvalue()


import docx2txt


def doc2text(file_path, doc_content):
    tmp_file_path = '/tmp/%d' % hash(file_path)
    with open(tmp_file_path, 'w') as ftmp:
        ftmp.write(doc_content)

    cmd = ['antiword', tmp_file_path]
    p = Popen(cmd, stdout=PIPE)
    stdout, stderr = p.communicate()
    return stdout


import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO


def pdf2text(pdf_content):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)

    fp = io.BytesIO(pdf_content)

    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(
            fp,
            pagenos,
            maxpages=maxpages,
            password=password,
            caching=caching,
            check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


#------------------------------------------------------------------------------------------

# ## path to files

#------------------------------------------------------------------------------------------

files_paths = []
for root, dirs, files in os.walk("../data/"):
    path = root.split('/')

    for f in files:
        if "DS_Store" not in f:
            path_to_file = path + [f]
            files_paths.append('/'.join(path_to_file))

#------------------------------------------------------------------------------------------

import magic

#------------------------------------------------------------------------------------------

from collections import defaultdict

#------------------------------------------------------------------------------------------

import urllib

# ## Detect file doctype

#------------------------------------------------------------------------------------------

#         file_doctype = magic.from_buffer(fin.read(10000))

#------------------------------------------------------------------------------------------

files_extentions = []
files_magic = []
files_urls = []

# for f in files_paths:


def get_file_info(f):
    with open(f, 'r') as fin:
        first_line = urllib.unquote(fin.readline())
        filename, file_extension = os.path.splitext(first_line.strip())

        if file_extension:
            file_extension = file_extension.lower().split('?')[0].strip(
            ).split('&')[0].strip()
            if "htm" in file_extension:
                file_extension = ".html"
            if "php" in file_extension:
                file_extension = ".php"

#         files_extentions.append(file_extension)

#         files_urls.append(first_line)

        file_magic = magic.from_buffer(fin.read(10000))
        #         files_magic.append(file_magic)

        return first_line, f, file_extension, file_magic


#------------------------------------------------------------------------------------------

import multiprocessing as mp
p = mp.Pool()

files_info = p.imap(get_file_info, files_paths)

#------------------------------------------------------------------------------------------

import pandas as pd

#------------------------------------------------------------------------------------------

files_df = pd.DataFrame(
    #     zip(files_urls, files_paths, files_extentions, files_magic),
    list(files_info),
    columns=["url", "path", "extention", "magic"])
files_df.loc[:, "doctype"] = files_df.magic.str.split(',').apply(
    lambda x: x[0])
files_df = files_df.set_index("path")

#------------------------------------------------------------------------------------------

print(files_df.shape)

#------------------------------------------------------------------------------------------

p.close()

# # Check processed files

#------------------------------------------------------------------------------------------

from os import listdir
from os.path import isfile, join
onlyfiles = [
    f for f in listdir("../processed/") if isfile(join("../processed/", f))
]
processed = [
    '../data/' +
    '/'.join(f.replace('html-', '').replace('txt-', '').split('_'))
    for f in onlyfiles
]
files_df.loc[files_df.index.isin(processed), "is_processed"] = True

#------------------------------------------------------------------------------------------

files_df[files_df.is_processed != True]

# ### Parse Files

#------------------------------------------------------------------------------------------

import sys
from bs4 import BeautifulSoup
import zlib

#------------------------------------------------------------------------------------------

not_in_parser = [
    'zlib compressed data',
    'Composite Document File V2 Document',
    'data',
]

bs4_parsable_doctypes = [
    'HTML document', 'XML 1.0 document', 'XML 1.0 document text',
    'ISO-8859 text', 'UTF-8 Unicode text', 'very short file (no magic)',
    'XML 1"  document', 'ASCII text', 'UTF-8 Unicode (with BOM) text',
    'PHP script'
]

pdf2text_doctypes = ['PDF document']

rtf2text_doctypes = ['MIME entity', 'Rich Text Format data']

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------


def st0_gen_data_flow(files_df):
    for file_path, file_meta in files_df.iterrows():
        #         if file_path == "../data/10/949496214207617953":
        if file_meta.is_processed != True:
            yield file_path, file_meta


#------------------------------------------------------------------------------------------


def st1_read_file(files_flow):
    for file_path, file_meta in files_flow:

        with open(file_path) as fin:
            fin.readline()
            file_content = fin.read()

        yield file_path, file_meta, file_content


#------------------------------------------------------------------------------------------

import sys

#------------------------------------------------------------------------------------------


def st2_parse_file(files_flow_with_content_obj):
    for file_path, file_meta, file_content in files_flow_with_content:
        try:
            if file_meta.doctype in pdf2text_doctypes or file_meta.extention == ".pdf":
                yield file_path, file_meta, pdf2text(file_content)

            elif file_meta.extention == ".doc":
                yield file_path, file_meta, doc2text(file_content)

            elif file_meta.doctype in rtf2text_doctypes:
                yield file_path, file_meta, rtf2text(file_content)

            elif file_meta.doctype in bs4_parsable_doctypes:
                file_bs = BeautifulSoup(file_content, "html5lib")
                yield file_path, file_meta, file_bs
            else:
                file_bs = BeautifulSoup(file_content, "html5lib")
                yield file_path, file_meta, file_bs
        except:
            msg1 = sys.exc_info()[0]
            try:
                file_bs = BeautifulSoup(file_content, "html5lib")
                yield file_path, file_meta, file_bs
            except:
                msg2 = sys.exc_info()[0]
                print(file_path, file_meta.doctype, file_meta.extention, msg1,
                      msg2)
                errors[file_path] = [msg1, msg2]


#------------------------------------------------------------------------------------------

import io


def st3_save_parsed_file(files_flow_parsed):
    for file_path, file_meta, file_parsed in files_flow_parsed:
        if isinstance(file_parsed, BeautifulSoup):
            file_type = "html"
            try:
                file_text = unicode(file_parsed).encode('utf-8')
            except RuntimeError as e:
                print("RuntimeError", file_path)
                file_text = unicode(file_parsed.get_text()).encode('utf-8')
        else:
            file_type = "txt"
            file_text = file_parsed

        path_to_parsed_text = ("../processed/%s-%s_%s" % (
            file_type, file_path.split('/')[-2], file_path.split('/')[-1]))
        file_text_compressed = zlib.compress(file_text)

        with open(path_to_parsed_text, "wb") as myfile:
            myfile.write(file_text_compressed)

        yield file_path, file_meta, path_to_parsed_text


#------------------------------------------------------------------------------------------

# errors = {}
# not_precessed = []

# files_flow = st0_gen_data_flow(files_df)
# files_flow_with_content = st1_read_file(files_flow)
# files_flow_parsed = st2_parse_file(files_flow_with_content)
# files_flow_saved = st3_save_parsed_file(files_flow_parsed)
# # files_flow_with_parts = st3_extract_document_parts(files_flow_parsed)
# # files_main_len = get_main_part_len(files_flow_with_parts)

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

# for i, (file_path, file_meta, path_to_parsed_text) in enumerate(files_flow_saved):
#     sys.stdout.write('\r' + "%s | %s" % (i, file_path))

#------------------------------------------------------------------------------------------

# assert(False)

#------------------------------------------------------------------------------------------

# files_df.is_processed

#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------

# len(to_process)

#------------------------------------------------------------------------------------------

from bs4 import UnicodeDammit

#------------------------------------------------------------------------------------------


def parse_file(file_data):
    file_path, file_meta = file_data

    with open(file_path) as fin:
        fin.readline()
        file_content = fin.read()

    try:
        if file_meta.doctype in pdf2text_doctypes or file_meta.extention == ".pdf":
            file_bs = pdf2text(file_content)
        elif file_meta.extention == ".doc":
            file_bs = doc2text(file_content)
        elif file_meta.doctype in rtf2text_doctypes:
            file_bs = rtf2text(file_content)
        else:  # file_meta.doctype in bs4_parsable_doctypes:
            file_bs = BeautifulSoup(file_content, "html5lib")
            file_bs = file_bs
    except:
        msg1 = sys.exc_info()[0]
        file_bs = BeautifulSoup(file_content, "html5lib")

    file_parsed = file_bs

    if isinstance(file_parsed, BeautifulSoup):
        file_type = "html"
        try:
            file_text = unicode(file_parsed).encode('utf-8')
        except RuntimeError as e:
            print("RuntimeError", file_path)
            file_text = unicode(file_parsed.get_text()).encode('utf-8')
    else:
        file_type = "txt"
        file_text = file_parsed

    path_to_parsed_text = ("../processed/%s-%s_%s" % (
        file_type, file_path.split('/')[-2], file_path.split('/')[-1]))
    #     path_to_parsed_text = "/tmp/parsed"
    file_text_compressed = zlib.compress(file_text)

    with open(path_to_parsed_text, "wb") as myfile:
        myfile.write(file_text_compressed)

    return file_path, file_meta, path_to_parsed_text


#------------------------------------------------------------------------------------------

to_process = [(file_path, file_meta)
              for file_path, file_meta in files_df.iterrows()
              if file_meta.is_processed != True]

#------------------------------------------------------------------------------------------

print(len(to_process))

#------------------------------------------------------------------------------------------

import multiprocessing

#------------------------------------------------------------------------------------------

p1 = multiprocessing.Pool(processes=8)

mapper_gen = p1.imap_unordered(parse_file, to_process, chunksize=1)

#------------------------------------------------------------------------------------------

for i, (file_path, file_meta, path_to_parsed_text) in enumerate(mapper_gen):
    sys.stdout.write('\r' + "%s | %s" % (i, file_path))

#------------------------------------------------------------------------------------------

p1.close()

#------------------------------------------------------------------------------------------
