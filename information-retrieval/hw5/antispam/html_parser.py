from collections import defaultdict

from HTMLParser import HTMLParser


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
