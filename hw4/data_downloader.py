import twitter
import json
import time
import sys
from collections import defaultdict
import pymongo
from .privat_constatns import (CONSUMER_KEY, CONSUMER_SECRET,
                               ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
from .data_store import store_twitts, clean_statuses, update_collection
from .constants import RLE_ERROR_CODE, DEBUG


def get_user_timline_rate_limit(api):
    return api.GetRateLimitStatus('statuses')[u'resources'][u'statuses']['/statuses/user_timeline']


def get_user_timeline(api,
                      user_id,
                      max_id=None,
                      since_id=None,
                      **kwargs):
    """https://dev.twitter.com/rest/public/timelines"""

    request_params = {
        'user_id': user_id,
        'screen_name': kwargs.pop("screen_name", None),
        'count': kwargs.pop("count", 200),
        'include_rts': kwargs.pop("include_rts", True),
        'trim_user': kwargs.pop("trim_user", True),
        'exclude_replies': kwargs.pop("exclude_replies", True),
    }

    error = {}

    try:
        statuses = api.GetUserTimeline(max_id=max_id,
                                       since_id=since_id,
                                       **request_params)
    except twitter.TwitterError as e:
        statuses = []

        if isinstance(e.message, list):
            if e.message[0].get("code") == RLE_ERROR_CODE:
                error = {"message": e.message[0]}
                sleep_in_seconds = api.GetSleepTime('statuses/user_timeline')

                if DEBUG:
                    for i in xrange(sleep_in_seconds, 0, -1):
                        time.sleep(1)
                        sys.stdout.write('\r' + "Sleep for %s seconds.." % i)
                        sys.stdout.flush()
        elif isinstance(e.message, basestring):
            if e.message == "Not authorized.":
                error = {"message": e.message}
                if DEBUG:
                    print "\nNot authorized for user with id=%s\n" % user_id

    return statuses, error


def statuses_downloader(api, user_id, **kwargs):
    max_id = None
    since_id = None

    is_greedy = kwargs.pop('is_greedy')

    while True:
        statuses, error = get_user_timeline(api, user_id,
                                            max_id, since_id, **kwargs)

        if error.get("code") == RLE_ERROR_CODE:
            continue

        if statuses:
            max_id = statuses[-1].id - 1
        else:
            raise StopIteration()

        yield statuses

        if not is_greedy: break


def download_twitts(users_ids, sphere_db=None):
    api = twitter.Api(consumer_key=CONSUMER_KEY,
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)

    statuses_downloader_params = {
        "count": 200,
        "include_rts": True,
        "trim_user": True,
        "exclude_replies": True,
        "is_greedy": False
    }

    if not sphere_db:
        sts = defaultdict(list)

    # [2874262145] - protected id, for debugging
    for user_i, user_id in enumerate(users_ids):
        if not sphere_db:
            if user_id in sts: continue

        user_statuses_downloader = statuses_downloader(api, user_id,
                                                       **statuses_downloader_params)

        for statuses_i, statuses in enumerate(user_statuses_downloader):
            statuses = [st.AsDict() for st in statuses]

            if statuses and sphere_db:
                statuses = clean_statuses(statuses)
                update_collection(sphere_db.twitts, statuses)
            else:
                sts[user_id] += statuses

            stats = {
                'user_i': user_i + 1,
                'statuses_i': statuses_i,
                'curr_sts_count': len(statuses)
            }
            sys.stdout.write(("\r{user_i} | " + \
                              "statuses_i={statuses_i} curr_sts_count={curr_sts_count}").format(**stats))
            sys.stdout.flush()

    if not sphere_db:
        store_twitts(sts)
