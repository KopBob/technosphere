# -*- encoding: utf-8 -*-


from collections import Counter

import re
from nltk.tokenize import TweetTokenizer, RegexpTokenizer


try:
    # Wide UCS-4 build
    emoji_re = re.compile(u'['
        u'\U0001F300-\U0001F64F'
        u'\U0001F680-\U0001F6FF'
        u'\u2600-\u26FF\u2700-\u27BF]+',
        re.UNICODE)
except re.error:
    # Narrow UCS-2 build
    emoji_re = re.compile(u'('
        u'\ud83c[\udf00-\udfff]|'
        u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
        u'[\u2600-\u26FF\u2700-\u27BF])+',
        re.UNICODE)


tokenizer = RegexpTokenizer(r'\w+')
tweet_tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=False)
hashtag_re = re.compile(r'(#[\w]+)')
url_re = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))')
retweet_re = re.compile(r'(RT @(?:\b\w+)+)')
numbers_re = re.compile(r"[ \n\r][-\d.]+[ \n\r!\\\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]")



def get_emoji(text):
    text = url_re.sub('', text)
    return emoji_re.sub('', text)


def get_hashtags(text):
    text = url_re.sub('', text)
    return hashtag_re.findall(text)

def get_words(text):
    """returns list of words"""
#     print repr(text)
    text = url_re.sub('', text)

    emojis = emoji_re.findall(text)
    text = emoji_re.sub('', text)

    hashtags = hashtag_re.findall(text)
    text = hashtag_re.sub('', text)

    retweets = [w.replace(" ", '') for w in retweet_re.findall(text)]
    text = retweet_re.sub('', text)

    numbers = numbers_re.findall(text)
    text = numbers_re.sub('', text)

    words = [word.lower() for word in tokenizer.tokenize(text)]

    return emojis + hashtags + words


from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import string

wnl = WordNetLemmatizer()
ss = SnowballStemmer('english')

def get_tokens(words):
    """returns list of tokens"""
    words = [ss.stem(token) for token in words]
    words = [wnl.lemmatize(token) for token in words]

    filtered_words = [word for word in words if (word not in stopwords.words('english')) and (len(word) > 2)]

    return filtered_words


def tweets2words(tweets):
    words = []
    for tweet in tweets:
        words += get_words(tweet)
    return words

def tweets2tokens(tweets):
    return get_tokens(tweets2words(tweets))


def tweets2counter(tweets):
    return Counter(tweets2tokens(tweets))