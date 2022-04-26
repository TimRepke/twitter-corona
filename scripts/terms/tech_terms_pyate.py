from pyate import combo_basic
import json
import numpy as np

FILE_TOPICS = 'data/geoengineering/topics.npy'
FILE_TWEETS = 'data/geoengineering/tweets_classified2.jsonl'

print('Reading topics...')
with open(FILE_TOPICS, 'rb') as f:
    TOPICS = np.load(f)

with open('data/climate2/tweets_filtered_10000.jsonl') as f:
    texts = [json.loads(line)['clean_text'] for line in f]

# texts = [t['text'] for t in tweets[:1000]]
texts = [t['clean_text'] for t in tweets[:10000]]

kws = combo_basic(texts, verbose=True)

print(kws.sort_values(ascending=False))
