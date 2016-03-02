import collections

import pymongo

from lib.nlp import tweets2words, tweets2tokens, tweets2counter

client = pymongo.MongoClient('mongodb://localhost:27117/')

db = client.sphere_kaggle


def get_tweets_grouped_by_user_cursor(collection, users_ids, include_retweets=False, limit=False):
    pipeline = []

    if not include_retweets:
        pipeline.append({
            "$match": {"retweeted_status": None}  # {"$ne": None}}
        })
    pipeline.append({
        "$match": {"user.id": {"$in": users_ids}}
    })
    pipeline.append({"$group": {
        "_id": "$user.id",
        "texts": {"$push": "$text"}
    }})
    pipeline.append({"$sort": {"id": -1}})
    #     if limit:
    #         pipeline.append({"$limit": limit})
    return collection.aggregate(pipeline, allowDiskUse=True)


def tweets_for_label(label):
    labeled_users_ids = db.users.distinct("id", {"label": label})
    tweets_cursor = get_tweets_grouped_by_user_cursor(collection=db.train_tweets,
                                                      users_ids=labeled_users_ids,
                                                      include_retweets=True)
    return tweets_cursor


def get_label_specific_counter(label):
    label_counter = collections.Counter({})

    for data in tweets_for_label(label):
        label_counter += tweets2counter(data["texts"])

    return label_counter


if __name__ == '__main__':
    label_2_counter = get_label_specific_counter(2)

    db.results.insert_one({
        "_id": "label_2_counter_rt",
        "data": dict(label_2_counter)
    })

    label_3_counter = get_label_specific_counter(3)

    db.results.insert_one({
        "_id": "label_3_counter_rt",
        "data": dict(label_3_counter)
    })
