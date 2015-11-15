import json
import pickle
import pymongo

from .constants import (DATABASE_URL, PATH_TO_USERS,
                        PATH_TO_USERS_IDS, PATH_TO_TWITTS)


def store_twitts(data, path=PATH_TO_TWITTS):
    print "\nDumping twitts to %s ..." % path
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def load_users_ids(path=PATH_TO_USERS_IDS):
    print "Reading users ids from %s ..." % path
    with open(path, 'rb') as infile:
        users_ids = json.load(infile)
        return users_ids


def load_twitts(path=PATH_TO_TWITTS):
    print "Reading users twitts from %s ..." % path
    with open(path, 'r') as infile:
        twitts = json.load(infile)
        return twitts


def load_users(path=PATH_TO_USERS):
    print "Reading users from %s ..." % path
    with open(path, 'rb') as infile:
        twitter_users = pickle.load(infile)
        return twitter_users


def clean_urls(obj):
    if "urls" not in obj:
        return obj

    for key, val in obj.get("urls").items():
        new_key = key.replace('t.co', 'tco')
        obj["urls"][new_key] = val

        del obj["urls"][key]

    return obj


def clean_users(users):
    cleaned_users = []

    for i, user in enumerate(users):
        user = user.AsDict()

        if "id" in user:
            user[u"_id"] = user.get("id")

        if "status" in user:
            del user["status"]

        cleaned_users.append(user)

    return cleaned_users


def clean_statuses(statuses):
    cleaned_statuses = []

    for status in statuses:
        if "id" in status:
            status[u"_id"] = status.get("id")

        if "urls" in status:
            status = clean_urls(status)

        if "retweeted_status" in status:
            status["retweeted_status"] = clean_urls(status["retweeted_status"])

        status[u"user_id"] = status["user"]
        cleaned_statuses.append(status)

    return cleaned_statuses


def clean_users_twitts(users_twitts):
    cleaned_twitts = []

    for user_id, twitts in users_twitts.items():
        for twitt in twitts:
            if "id" in twitt:
                twitt[u"_id"] = twitt.get("id")

            if "urls" in twitt:
                twitt = clean_urls(twitt)

            if "retweeted_status" in twitt:
                twitt["retweeted_status"] = clean_urls(twitt["retweeted_status"])

            twitt[u"user_id"] = user_id
            cleaned_twitts.append(twitt)

    return cleaned_twitts


def update_collection(collection, data):
    try:
        collection.insert(data, continue_on_error=True)
    except pymongo.errors.BulkWriteError as e:
        print json.dumps(e.details, indent=2)
    except pymongo.errors.DuplicateKeyError as e:
        print json.dumps(e.details, indent=2)


def drop_sphere_all(db):
    client = pymongo.MongoClient(DATABASE_URL)
    shpere_db = client.shpere

    shpere_db.twitts.drop()
    shpere_db.users.drop()


def update_database(users, twitts):
    client = pymongo.MongoClient(DATABASE_URL)
    shpere_db = client.shpere

    print "Clean users..."
    cleaned_users = clean_users(users)
    print "Insert users..."
    update_collection(shpere_db.users, cleaned_users)

    print "Clean twitts..."
    cleaned_twitts = clean_users_twitts(twitts)
    print "Insert twitts..."
    update_collection(shpere_db.twitts, cleaned_twitts)


def get_twitts_texts(db):
    return list(db.twitts.find({}, {'text': 1, '_id': 0}))
