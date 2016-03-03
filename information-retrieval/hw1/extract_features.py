#! /usr/env/bin python
# -*- encoding: utf-8

import re
import os

import linecache
import random
import urlparse
import urllib

from collections import defaultdict, Counter

threshold_filter = lambda counter, threshold: filter(lambda f: f[1] > threshold, counter.most_common())
len_filter = lambda array: filter(len, array)
get_ext = lambda x: os.path.splitext(x)[1][1:]
num_inside = lambda x: True if re.match(r"\D+\d+\D+", x) else False
unquote = lambda segments: map(urllib.unquote_plus, segments)


def lod2dol(lod):
    """list_of_dicts_to_dict_of_lists"""
    result = defaultdict(list)
    for d in lod:
        for k, v in d.items():
            result[k].append(v)
    return result


cache = defaultdict(dict)


class FeatureExtractor(object):
    def __init__(self, input_file_1_path, input_file_2_path, output_file_path):
        self.input_file_1_path = input_file_1_path
        self.input_file_2_path = input_file_2_path
        self.output_file_path = output_file_path

    # segment_ext_substr[0-9]_<index>:<extension value>
    @staticmethod
    def segment_ext_substr_num_index(urls, threshold=100):
        segs = [dict(enumerate(unquote(len_filter(url.path.split('/'))))) for url in urls]

        seg_by_index = lod2dol(segs)

        func = lambda x: (num_inside(x), get_ext(x))
        ext_len_filter = lambda x: len(x[1])

        t = {k: Counter(filter(ext_len_filter, map(func, seg_by_index[k]))) for k in seg_by_index.keys()}

        result = []

        for seg, counter in t.items():
            for (flag, ext), count in counter.items():
                if flag:
                    if count > threshold:
                        result.append(("segment_ext_substr[0-9]_%d:%s" % (seg, ext), count))

        return result

    # segment_substr[0-9]_<index>:1
    @staticmethod
    def segment_substr_num_index(urls, threshold=100):
        segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]

        seg_by_index = lod2dol(segs)
        func = lambda c: c[True] if c[True] > c[False] else 0

        t = {k: Counter(map(num_inside, unquote(seg_by_index[k])))[True] for k in seg_by_index.keys()}

        substr_num_segs = {k: v for k, v in t.items() if v > threshold}

        result = [("segment_substr[0-9]_%d:1" % (seg), count) for seg, count in substr_num_segs.items()]

        return result

    # segment_ext_<index>:<extension value>
    @staticmethod
    def segment_ext_index(urls, threshold=100):
        segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]

        seg_by_index = lod2dol(segs)

        t = {k: threshold_filter(Counter(len_filter(map(get_ext, seg_by_index[k]))), threshold) for k in
             seg_by_index.keys()}
        exts = {k: v for k, v in t.items() if len(v)}

        result = []

        for seg, ext_array in exts.items():
            for ext, count in ext_array:
                result.append(("segment_ext_%s:%s" % (seg, ext), count))

        return result

    # segment_[0-9]_<index>:1
    @staticmethod
    def extract_segment_is_number(urls, threshold=100):
        segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]

        seg_by_index = lod2dol(segs)

        isnum = lambda x: unicode(x).isnumeric()

        t = {k: Counter(map(isnum, seg_by_index[k]))[True] for k in seg_by_index.keys()}
        numeric_segs = {k: v for k, v in t.items() if v > threshold}

        result = [("segment_[0-9]_%d:1" % seg, count) for seg, count in numeric_segs.items()]

        return result

    # segment_len_<index>:<segment length>
    @staticmethod
    def extract_segment_len_index(urls, threshold=100):
        segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]

        seg_by_index = lod2dol(segs)

        t = {k: threshold_filter(Counter(map(len, seg_by_index[k])), threshold) for k in seg_by_index.keys()}

        result = []

        for seg, lengths in t.items():
            for length, count in lengths:
                result.append(("segment_len_%d:%d" % (seg, length), count))

        return result

    # segment_name_<index>:<string>
    @staticmethod
    def extract_segment_name_index(urls, threshold=100):
        segs = [dict(enumerate(len_filter(url.path.split('/')))) for url in urls]
        seg_pos = lod2dol(segs)

        t = {k: threshold_filter(Counter(seg_pos[k]), threshold) for k in seg_pos.keys()}

        result = []

        for seg, names in t.items():
            for name, count in names:
                result.append(("segment_name_%d:%s" % (seg, name), count))

        return result

    # param:<parameters=value>
    @staticmethod
    def extract_param(urls, threshold=100):
        queries = sum([len_filter(url.query.split('&')) for url in urls], [])
        queries_counter = Counter(queries)

        queries_cleaned = threshold_filter(queries_counter, threshold)

        result = [("param:%s" % q, c) for q, c in queries_cleaned]

        return result

    # param_name:<имя>
    @staticmethod
    def extract_param_name(urls, threshold=100):
        param_names = sum([urlparse.parse_qs(url.query).keys() for url in urls], [])
        param_names_counter = Counter(param_names)
        params_cleaned = threshold_filter(param_names_counter, threshold)

        result = [("param_name:%s" % p, c) for p, c in params_cleaned]

        return result

    # segments:<len>
    @staticmethod
    def extract_segments_len(urls, threshold=100):
        count_sg = lambda url: len(len_filter(url.path.split('/')))
        sg_counter = Counter([count_sg(url) for url in urls])
        sg_counts_cleaned = threshold_filter(sg_counter, threshold)

        result = [("segments:%d" % sg, count) for sg, count in sg_counts_cleaned]

        return result

    def get_sampled_lines(self, file_path, size=1000):
        # count lines in file
        if "n_lines" not in cache[file_path].keys():
            with open(file_path, 'r') as fp:
                cache[file_path]["n_lines"] = sum(1 for line in fp if line.rstrip())

        if size > cache[file_path]["n_lines"]:
            raise IOError("Sample size is to big")

        # get 1k sample
        sample = random.sample(range(cache[file_path]["n_lines"]), size)

        # read sampled lines
        sampled_lines = [linecache.getline(file_path, i + 1) for i in sample]

        return sampled_lines

    def extract_features(self):
        lines_1 = self.get_sampled_lines(self.input_file_1_path)
        lines_2 = self.get_sampled_lines(self.input_file_2_path)

        urls = [urlparse.urlparse(line.rstrip()) for line in lines_1 + lines_2]

        result = []

        result += self.extract_segments_len(urls)
        result += self.extract_param_name(urls)
        result += self.extract_param(urls)
        result += self.extract_segment_name_index(urls)
        result += self.extract_segment_len_index(urls)
        result += self.extract_segment_is_number(urls)
        result += self.segment_ext_index(urls)
        result += self.segment_substr_num_index(urls)
        result += self.segment_ext_substr_num_index(urls)

        result = sorted(result, key=lambda tup: tup[1], reverse=True)
        result_cleaned = ["%s\t%s\n" % r for r in result]

        with open(self.output_file_path, "w") as output:
            for item in result_cleaned:
                output.write(item)


def extract_features(INPUT_FILE_1, INPUT_FILE_2, OUTPUT_FILE):
    extractor = FeatureExtractor(INPUT_FILE_1, INPUT_FILE_2, OUTPUT_FILE)
    extractor.extract_features()
