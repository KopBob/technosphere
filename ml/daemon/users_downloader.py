import csv
import math
from bson.int64 import Int64


def get_user_ids_from_csv(path_to_file):
    with open(path_to_file, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        return [r[0] for r in reader]


def convert_ids_to_int64(ids):
    return [Int64(id) for id in ids]


def yield_chunks(array, n):
    """Yield successive n-sized chunks from array."""
    for i in xrange(0, len(array), n):
        yield array[i:i+n]


class UserDonwloader(object):
    CHUNK_SIZE = 100

    def __init__(self, api, collection):
        self.api = api
        self.collection = collection

    def download(self, user_ids):
        user_ids = convert_ids_to_int64(user_ids)
        missed_ids = self._get_missed_ids_in_db(user_ids)

        remain = len(missed_ids)
        for ids_chunk in yield_chunks(missed_ids, self.CHUNK_SIZE):
            users = self.api.lookup_users(user_ids=ids_chunk)
            users_json = [u._json for u in users]

            self.collection.insert_many(users_json)

            remain -= len(ids_chunk)
            print("Remain %d" % remain)

    def _get_missed_ids_in_db(self, new_ids):
        c = self.collection.aggregate([
            {"$match": {"id": {"$in": new_ids}}},
            {"$project": {"_id": "$id"}},
        ])

        existent = set([obj['_id'] for obj in c])
        missed = set(new_ids) - set(existent)

        return list(missed)
