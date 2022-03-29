from pyate import combo_basic
import json

with open('data/climate2/tweets_filtered_10000.jsonl') as f:
    tweets = [json.loads(l) for l in f]

# texts = [t['text'] for t in tweets[:1000]]
texts = [t['clean_text'] for t in tweets[:10000]]

kws = combo_basic(texts, verbose=True)

print(kws.sort_values(ascending=False))
