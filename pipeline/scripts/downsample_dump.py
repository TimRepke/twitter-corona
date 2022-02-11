import json
import random
import numpy as np

print('reading dump')
with open('data/climate2/topics_big/dump_7000000_monthly.json', 'r') as f:
    dump = json.load(f)

print('reading labels')
labels = np.load('data/climate2/topics_big/labels_7000000_tsne.npy')

print('selecting tweets')
downsampled_tweets = []
for topic in np.unique(labels):
    tweet_idxs = np.argwhere(labels == topic).reshape(-1, )
    np.random.shuffle(tweet_idxs)

    for i in tweet_idxs[:20]:
        downsampled_tweets.append(dump['tweets'][i])

dump['tweets'] = downsampled_tweets

print('dumping smaller dump')
with open('data/climate2/topics_big/dump_7000000_monthly_downsampled.json', 'w') as f:
    json.dump(dump, f)
