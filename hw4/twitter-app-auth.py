import json
from application_only_auth import Client

# The consumer secret is an example and will not work for real requests
# To register an app visit https://dev.twitter.com/apps/new
CONSUMER_KEY = 'QgnRAOglaJ6I0ulrIgP3R1mrt'
CONSUMER_SECRET = 'JrsmPHBodqeN8R9jbtyZEVbwGFMtWRBToLTjyCca2M33Rg5MYX'

client = Client(CONSUMER_KEY, CONSUMER_SECRET)

# Pretty print of tweet payload
tweet = client.request(url='https://api.twitter.com/1.1/statuses/user_timeline.json?screen_name=twitterapi&count=200')
# print json.dumps(tweet, sort_keys=True, indent=4, separators=(',', ':'))

# Show rate limit status for this application
status = client.rate_limit_status()
print status['resources']['statuses']['/statuses/user_timeline']