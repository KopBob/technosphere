#!/usr/bin/env python

from __future__ import print_function

import logging
import time
import csv
import json
import tweepy
import pymongo
import bson

from optparse import OptionParser

from .private_constants import CONSUMER_KEY, CONSUMER_SECRET, DB_URL

logger = logging.getLogger('log')


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bson.ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


def save_json(data, path):
    print("\nDumping to %s ..." % path)
    with open(path, 'w+') as outfile:
        json.dump(json.loads(JSONEncoder().encode(data)), outfile, indent=4)


def user_ids_from_csv(path_to_file):
    array = []
    with open(path_to_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        array = [int(r[0]) for r in reader]

    return array


def convert_ids_to_int64(ids):
    return [bson.int64.Int64(id) for id in ids]


class TweetsDownloader(object):
    def __init__(self, api, tweets_collection, users_collection,
                 errors_collection, **kwargs):
        self.api = api

        self.tweets_collection = tweets_collection
        self.users_collection = users_collection
        self.errors_collection = errors_collection

        self.pages_count = kwargs.get('pages_count', 5)
        self.tweets_threshold = kwargs.get('tweets_threshold', 500)

        self.api_settings = {
            'count': kwargs.get('count', 200),
            'include_rts': kwargs.get('include_rts', True),
            'trim_user': kwargs.get('trim_user', True),
            'exclude_replies': kwargs.get('exclude_replies', True)
        }

        self._cached_users_info = {}

        self.__stamp = int(time.time())

    @property
    def users_info(self):
        if self._cached_users_info:
            return self._cached_users_info

        self._cached_users_info = self._download_latest_tweets()
        return self._cached_users_info

    def download(self, user_ids):
        user_ids_for_downloading = self._filter_user_ids(convert_ids_to_int64(user_ids))

        logger.info("Start downloader %d", self.__stamp)
        logger.info("Will load %d from %d" % (len(user_ids_for_downloading), len(user_ids)))
        for i, user_id in enumerate(user_ids_for_downloading):
            user_logger = logging.getLogger('twitter_daemon.%d' % user_id)
            user_logger.setLevel(level=logging.DEBUG)

            user_info = self.users_info.get(user_id, None)
            max_id = None

            if user_info:
                if user_info.get("tweets_count") > self.tweets_threshold:
                    self._log_error(Exception("Tweets limit exceeded"), {"user_id": user_id})
                    continue

                max_id = user_info.get("last_tweet_id") - 1 \
                    if user_info.get("last_tweet_id") else None

            user_logger.debug("start loading user")
            try:
                cursor = tweepy.Cursor(self.api.user_timeline,
                                       user_id=user_id, max_id=max_id, **self.api_settings)
                for j, page in enumerate(cursor.pages(self.pages_count)):
                    user_logger.debug("(%d) |(%d) %d/%d", i + 1, len(page), j + 1, self.pages_count)

                    tweets_json = [t._json for t in list(page)]
                    try:
                        self.tweets_collection.insert_many(tweets_json)
                    except pymongo.errors.PyMongoError as e:
                        self._log_error(e, {"tweets": tweets_json})
            except tweepy.error.TweepError as e:
                self._log_error(e, {"user_id": user_id})

    def _log_error(self, exception, data, **kwargs):
        logger.error(exception)

        error_entity = {"stamp": self.__stamp}

        if issubclass(type(exception), pymongo.errors.PyMongoError):
            error_entity.update({"type": "mongo"})
        elif issubclass(type(exception), tweepy.error.TweepError):
            error_entity.update({"type": "twitter"})
        else:
            error_entity.update({"type": "none"})

        error_entity.update({
            "msg": repr(exception),
            "data": data
        })

        self.errors_collection.insert(error_entity)

    def _download_latest_tweets(self):
        users_info = {}

        cursor = self.tweets_collection.aggregate([
            {"$project": {"id": "$id", "user_id": "$user.id"}},
            {"$sort": {"id": -1}},
            {"$group": {
                "_id": "$user_id",
                "last_tweet_id": {"$last": "$id"},
                "tweets_count": {"$sum": 1}
            }},
        ], allowDiskUse=True)

        for t in cursor:
            u_id = t.get("_id").get('id') if isinstance(t.get("_id"), dict) \
                else t.get("_id")
            users_info[u_id] = {
                "last_tweet_id": t["last_tweet_id"],
                "tweets_count": t["tweets_count"]
            }

        return users_info

    def _filter_user_ids(self, user_ids):
        users_in_db_cursor = self.tweets_collection.aggregate([
            {"$match": {"user.id": {"$in": user_ids}}},
            {"$project": {"id": "$id", "user_id": "$user.id"}},
            {"$sort": {"id": -1}},
            {"$group": {
                "_id": "$user_id",
                "last_tweet_id": {"$last": "$id"},
                "tweets_count": {"$sum": 1}
            }},
        ], allowDiskUse=True)

        users_info_cursor = self.users_collection.aggregate([
            {"$match": {"id": {"$in": user_ids}}},
            {"$project": {"_id": "$id", "statuses_count": "$statuses_count"}},
        ], allowDiskUse=True)

        users_in_db = list(users_in_db_cursor)
        users_info = list(users_info_cursor)

        users_in_db_dict = {u["_id"]: {"last_tweet_id": u["last_tweet_id"], "tweets_count": u["tweets_count"]} for u in
                            users_in_db}
        self._cached_users_info = users_in_db_dict

        users_info_dict = {u["_id"]: {"statuses_count": u["statuses_count"]} for u in users_info}

        target_arr = []

        for id in user_ids:
            if id not in users_in_db_dict.keys():
                target_arr.append(id)
                continue

            if id not in users_info_dict.keys():
                print("Unknown user", id)
                target_arr.append(id)
                continue

            current = users_in_db_dict.get(id).get('tweets_count')
            potential = users_info_dict.get(id).get('statuses_count')

            if current / 200 == potential / 200:
                continue

            if current < potential:
                if current < self.tweets_threshold:
                    target_arr.append(id)
                    continue

        return target_arr


def run(settings):
    auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    db = pymongo.MongoClient(settings.get("db_url"))

    logger.info("Getting user ids..")
    user_ids = user_ids_from_csv(settings.get("csv_file"))

    tweets_collection = db.sphere_kaggle[settings.get("tweets_collection")]
    users_collection = db.sphere_kaggle[settings.get("users_collection")]
    errors_collection = db.sphere_kaggle[settings.get("errors_collection")]

    tweets_collection.create_index('id', unique=True)
    users_collection.create_index('id', unique=True)

    downloader = TweetsDownloader(api=api,
                                  tweets_collection=tweets_collection,
                                  users_collection=users_collection,
                                  errors_collection=errors_collection)
    downloader.download(user_ids)


SETTINGS = {
    "debug": {
        "csv_file": "./files/debug.csv",
        "tweets_collection": "debug"
    },
    "test": {
        "csv_file": "./files/twitter_test.csv",
        "tweets_collection": "test_tweets"
    },
    "train": {
        "csv_file": "./files/twitter_train.csv",
        "tweets_collection": "train_tweets"
    }
}

DEFAULT_SETTINGS = {
    "db_url": DB_URL,
    "errors_collection": "errors",
    "users_collection": "users"
}

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR,
                        format='%(asctime)s %(levelname)s\t%(name)s %(funcName)s\t%(message)s')
    logger.setLevel(level=logging.DEBUG)

    parser = OptionParser()
    parser.add_option("-d", "--dataset", dest="dataset",
                      help=("dataset for downloading (%s)" % "/".join(SETTINGS.keys())))

    (options, args) = parser.parse_args()

    if not options.dataset:
        parser.error('dataset not given')
    elif options.dataset not in SETTINGS.keys():
        parser.error('%s - wrong dataset' % options.dataset)

    app_settings = {}
    app_settings.update(DEFAULT_SETTINGS)
    app_settings.update(SETTINGS.get(options.dataset))

    logger.info(app_settings)
    run(app_settings)
