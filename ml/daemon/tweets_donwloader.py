#!/usr/bin/env python

from __future__ import print_function

import logging
import sys
import csv
import json
import tweepy
import pymongo

from collections import defaultdict

from bson import ObjectId

logger = logging.getLogger('log')

CONSUMER_KEY = "wTOj2975uKAhyIyJkrjod2wju"
CONSUMER_SECRET = "v0IpKZ6LxnUN0DgGGAmLE7tgzjiK6u5zMYZnSIFX5ghN8hUoTp"


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
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


class TweetsDownloader(object):
    LOG_DATA_FILE = "./files/twt_downloader_log_data.json"

    def __init__(self, collection, **kwargs):
        self.collection = collection
        self.pages_count = kwargs.get('pages_count', 5)
        self.tweets_threshold = kwargs.get('tweets_threshold', 500)

        self.api_settings = {
            'count': kwargs.get('count', 200),
            'include_rts': kwargs.get('include_rts', True),
            'trim_user': kwargs.get('trim_user', True),
            'exclude_replies': kwargs.get('exclude_replies', True)
        }

        self._cached_users_info = {}
        self._failed_tweets = defaultdict(list)
        self._protected_users = []

    @property
    def users_info(self):
        if self._cached_users_info:
            return self._cached_users_info

        self._cached_users_info = self._download_latest_tweets()
        return self._cached_users_info

    def download(self, user_ids):
        for i, user_id in enumerate(user_ids):
            logger = logging.getLogger('twitter_daemon.%d' % user_id)

            user_info = self.users_info.get(user_id, None)
            max_id = None

            if user_info:
                if user_info.get("tweets_count") > self.tweets_threshold:
                    logger.warn("To mutch tweets.")
                    continue

                max_id = user_info.get("last_tweet_id") - 1 \
                    if user_info.get("last_tweet_id") else None

            try:
                cursor = tweepy.Cursor(api.user_timeline,
                                       user_id=user_id, max_id=max_id, **self.api_settings)

                logger.debug("start loading user %d", int(user_id))
                for j, page in enumerate(cursor.pages(self.pages_count)):
                    logger.debug("(%d) |(%d) %d/%d", i + 1, len(page), j + 1, self.pages_count)

                    tweets_json = [t._json for t in list(page)]
                    try:
                        self.collection.insert_many(tweets_json)
                    except pymongo.errors.BulkWriteError as e:
                        logger.error(e)
                        self._failed_tweets[user_id] += tweets_json
            except tweepy.error.TweepError as e:
                logger.error(e)
                if e.response.status_code == 401:
                    self._protected_users.append(user_id)
                    continue
                else:
                    raise e

        self._save_execution_data()

    def _download_latest_tweets(self):
        users_info = {}

        cursor = self.collection.aggregate([
            {"$project": {"id": "$id", "user_id": "$user.id"}},
            {"$sort": {"id": -1}},
            {"$group": {
                "_id": "$user_id",
                "last_tweet_id": {"$last": "$id"},
                "tweets_count": {"$sum": 1}
            }},
        ])

        for t in cursor:
            u_id = t.get("_id").get('id') if isinstance(t.get("_id"), dict) \
                else t.get("_id")
            users_info[u_id] = {
                "last_tweet_id": t["last_tweet_id"],
                "tweets_count": t["tweets_count"]
            }

        return users_info

    def _save_execution_data(self):
        logger.info('Saving log data')

        log_data = {
            "failed_tweets": self._failed_tweets,
            "protected_users": self._protected_users
        }
        save_json(log_data, self.LOG_DATA_FILE)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR,
                        format='%(levelname)s\t%(name)s\t%(funcName)s\t%(message)s')
    logger.setLevel(level=logging.DEBUG)

    auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    db = pymongo.MongoClient('mongodb://donkey:27017/')

    logger.info("Getting user ids..")
    user_ids = user_ids_from_csv("./files/twitter_test.csv")

    db.sphere_kaggle.test_tweets.create_index('id', unique=True)
    downloader = TweetsDownloader(db.sphere_kaggle.test_tweets)
    logger.info("Start downloading..")
    downloader.download(user_ids)
