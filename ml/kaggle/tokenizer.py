import pandas as pd
from lib import nlp

import pymongo


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


if __name__ == '__main__':
    client = pymongo.MongoClient('mongodb://localhost:27117/')
    db = client.sphere_kaggle
    db.train_text_data.create_index('twitter_id', unique=True)
    # {
    #     "twitter_id",
    #     "text"
    # }

    train_users = pd.read_csv("./daemon/files/twitter_train.csv")
    train_users_ids = list(train_users.twitter_id)

    train_users_cursor = get_tweets_grouped_by_user_cursor(collection=db.train_tweets,
                                                           users_ids=train_users_ids,
                                                           include_retweets=False)

    for i, train_user_data in enumerate(train_users_cursor):
        print("preparing %d user" % i)
        db.train_text_data.insert_one({
            "twitter_id": train_user_data.get("_id"),
            "text": " ".join(nlp.tweets2tokens(train_user_data.get("texts"))),
        })
