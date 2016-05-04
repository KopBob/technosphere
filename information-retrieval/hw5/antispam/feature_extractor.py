# coding: utf-8

import re
import base64
import zlib
from collections import defaultdict

from HTMLParser import HTMLParser
from nltk.stem import SnowballStemmer

from constants import RUS_STOP_WORDS, ESP, COMMON_TAGS, SPETIAL_TAGS


class SpamHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.__text = []

        self.data = defaultdict(list)
        self.curr_tag = None

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            attrs_dict = dict(attrs)
            if 'alt' in attrs_dict.keys():
                if attrs_dict.get('alt'):
                    self.data[self.curr_tag].append(attrs_dict.get('alt'))
        if tag == 'meta':
            if attrs:
                for k, v in attrs:
                    if v:
                        self.data[self.curr_tag].append(v)
        self.curr_tag = tag

    def handle_data(self, data):
        if data:
            self.data[self.curr_tag].append(data)

        self.__text.append(data)

    def handle_endtag(self, tag):
        self.curr_tag = None

    def text(self):
        return ' '.join(self.__text).strip()


class HTMLDataExtractor:
    def __init__(self, parser_class=SpamHTMLParser):
        self.parser = parser_class()
        self.stemmer = SnowballStemmer("russian")

        self.data = {
            "text": None,
            "tags_data": None
        }

    def clean_text(self, text):
        russian_words = re.findall(u"[\u0400-\u0500]+", text.lower())
        # cleaned_russian_words = filter(lambda x: len(x) > 2, russian_words)
        #
        # stemmed_words = [self.stemmer.stem(w) for w in cleaned_russian_words]
        cleaned_words = filter(lambda x: x not in RUS_STOP_WORDS, russian_words)

        cleaned_text = " ".join(cleaned_words)

        return cleaned_text

    def clean_tags_data(self, tags_data):
        cleaned_tags_data = {}

        for tag_name in tags_data.keys():
            cleaned_key_texts = []
            for text in tags_data[tag_name]:
                cleaned_text = self.clean_text(text)

                if len(cleaned_text) != 0:
                    cleaned_key_texts.append(cleaned_text)

            if len(cleaned_key_texts) != 0:
                cleaned_tags_data[tag_name] = cleaned_key_texts

        return cleaned_tags_data

    def extract(self, html):
        self.parser.feed(html)

        self.data["text"] = self.clean_text(self.parser.text())
        self.data["tags_data"] = self.clean_tags_data(self.parser.data)

        return self.data


class HTMLFeatureExtractor:
    """
        # ratio_of_total_num_of_tags_to_empty_tags
            # отношение кол-ва тегов к числу путсых тегов
        # ratio_of_words_count_in_tags_to_empty_tags
            # отношение кол-ва слов обернутых в теги к тексту вне тегов
        # ratio_of_words_count_to_unique_words_count
            # общее кол-во слов
        # ratio_of_total_text_to_compressed
            # отношение длины текста к его сжатой версии

        # @@ratio_of_special_tag_text_len_to_compressed
            # отношение длины текста в специальных тегах к его сжатой версии
        # @@
            # отношение длины текста в обычных тегах к его сжатой версии

        # ratio_of_ordinary_tags_count_to_special_tags
            # отношение числа обычных тегов к тегам для форматирования текста
            # big, dt, li, a, title, b, thead, h1, h2, h3, h4, h5, h6, style, header,
            # strong, aside, ol, small, , section, i, em, abbr, cite, ins, dfn
            # strike, pre, code, tt, blockquote, font,

        # NO [tag form common_tags]_count
            # кол-во рассматриваемого тега в документе
        # [tag form common_tags]_mean_words_count
            # среднее кол-во слов обернутых в рассматриваемом теге
        # [tag form common_tags]_ratio_of_total_to_unique_words_counts
            # отношение кол-ва всех слов к кол-ву уникальных в тегах рассматриваемого типа
        # [tag form common_tags]_ratio_of_total_text_to_compressed # len(zlib.compress(str1, 0)) / float(len(zlib.compress(str1)))
            # отношение длины конкат. текста к его сжатой версии для рассматриваемого тега
    """

    def __init__(self, common_tags=COMMON_TAGS):
        self.features = None
        self.data = defaultdict(dict)
        self.features = {}

    def extract(self, data):
        total_tags_count = 0
        total_words_count = 0
        special_tags_count = 0

        for tag_name, bodies in data["tags_data"].iteritems():
            joined_bodies = ' '.join(bodies)
            words = joined_bodies.split()
            unique_words = set(words)

            total_tags_count += len(bodies)
            total_words_count += len(words)
            special_tags_count += len(bodies) if tag_name in SPETIAL_TAGS else 0

            self.data[tag_name]["bodies"] = bodies
            self.data[tag_name]["words_count"] = len(words)

            # [tag form common_tags]_count
            self.data[tag_name]["count"] = len(bodies)

            # [tag form common_tags]_mean_words_count
            self.data[tag_name]["mean_words_count"] = \
                sum([len(body.split()) for body in bodies]) / (float(len(bodies)) + ESP)

            # [tag form common_tags]_ratio_of_total_words_to_unique
            self.data[tag_name]["ratio_of_total_to_unique_words_counts"] = \
                len(words) / (float(len(unique_words)) + ESP)

            # [tag form common_tags]_ratio_text_to_compressed
            self.data[tag_name]["ratio_of_total_text_to_compressed"] = \
                len(zlib.compress(joined_bodies.encode("utf-8"), 0)) / float(
                    len(zlib.compress(joined_bodies.encode("utf-8"))))

        for tag_name in SPETIAL_TAGS:
            if tag_name in data["tags_data"].keys():
                self.features["%s_count" % tag_name] = \
                    self.data[tag_name]["count"]
                self.features["%s_mean_words_count" % tag_name] = \
                    self.data[tag_name]["mean_words_count"]
                self.features["%s_ratio_of_total_to_unique_words_counts" % tag_name] = \
                    self.data[tag_name]["ratio_of_total_to_unique_words_counts"]
                self.features["%s_ratio_of_total_text_to_compressed" % tag_name] = \
                    self.data[tag_name]["ratio_of_total_text_to_compressed"]
            else:
                self.features["%s_count" % tag_name] = 0.0
                self.features["%s_mean_words_count" % tag_name] = 0.0
                self.features["%s_ratio_of_total_to_unique_words_counts" % tag_name] = 0.0
                self.features["%s_ratio_of_total_text_to_compressed" % tag_name] = 0.0

        text_words = data["text"].split()
        text_unique_words = set(text_words)

        # ratio_of_total_num_of_tags_to_empty_tags
        # отношение кол-ва тегов к числу путсых тегов
        empty_tag_count = float(self.data[None]["count"]) if "count" in self.data.keys() else 0.0
        self.features["ratio_of_total_num_of_tags_to_empty_tags"] = \
            (total_tags_count - empty_tag_count) / (empty_tag_count + ESP)

        # ratio_of_words_count_in_tags_to_empty_tags
        # отношение кол-ва слов обернутых в теги к тексту вне тегов
        empty_tag_words_count = float(self.data[None]["words_count"]) if "words_count" in self.data.keys() else 0.0
        self.features["ratio_of_words_count_in_tags_to_empty_tags"] = \
            (total_words_count - empty_tag_words_count) / (empty_tag_words_count + ESP)

        # ratio_of_words_count_to_unique_words_count
        self.features["ratio_of_words_count_to_unique_words_count"] = \
            len(text_words) / (float(len(text_unique_words)) + ESP)

        # ratio_of_total_text_to_compressed
        # отношение длины текста к его сжатой версии
        self.features["ratio_of_total_text_to_compressed"] = \
            len(zlib.compress(data["text"].encode("utf-8"), 0)) / float(
                len(zlib.compress(data["text"].encode("utf-8"))))

        # ratio_of_ordinary_tags_count_to_special_tags
        # отношение числа обычных тегов к тегам для форматирования текста

        self.features["ratio_of_ordinary_tags_count_to_special_tags"] = \
            (total_tags_count - empty_tag_count - special_tags_count) / float(
                special_tags_count + ESP)

        return self.data


def collect_html_features(pageInb64):
    html = base64.b64decode(pageInb64).decode('utf-8')

    feature_extractor = HTMLFeatureExtractor()
    feature_extractor.extract(html)
