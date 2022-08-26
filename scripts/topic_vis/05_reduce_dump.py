import json
import numpy as np

FULL_DUMP = 'data/geoengineering/topics_finn3/full_dump.json'
LABELS_FILE = 'data/geoengineering/topics_finn3.npy'
TARGET_FILE = 'data/geoengineering/topics_finn3/dump_geo_filtered.json'
MAX_TWEETS_PER_TOPIC = 500

print('reading dump')
with open(FULL_DUMP, 'r') as f:
    dump = json.load(f)

print('reading labels')
labels = np.load(LABELS_FILE)

print('selecting tweets')
downsampled_tweets = []
for topic in np.unique(labels):
    tweet_idxs = np.argwhere(labels == topic).reshape(-1, )
    np.random.shuffle(tweet_idxs)

    for i in tweet_idxs[:MAX_TWEETS_PER_TOPIC]:
        downsampled_tweets.append(dump['tweets'][i])

dump['tweets'] = downsampled_tweets

print('dumping smaller dump')
with open(TARGET_FILE, 'w') as f:
    json.dump(dump, f)
