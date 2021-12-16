import os.path
from datetime import datetime
import json
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from utils.tweets import clean_tweet, get_urls, get_hashtags, get_mentions
import math
import time


def process_tweet(tweet):
    tweet['clean_text'] = clean_tweet(tweet['text'],
                                      remove_hashtags=True,
                                      remove_urls=True,
                                      remove_mentions=True,
                                      remove_nonals=True)
    hashtags = get_hashtags(tweet['text'])
    urls = get_urls(tweet['text'])
    mentions = get_mentions(tweet['text'])
    n_tokens = len(tweet['clean_text'].split())
    tweet['meta'] = {
        'n_tokens': n_tokens,
        'n_tokens_raw': len(hashtags) + len(urls) + len(mentions) + n_tokens,
        'n_hashtags': len(hashtags),
        'hashtags': hashtags,
        'n_urls': len(urls),
        'urls': urls,
        'n_mentions': len(mentions),
        'mentions': mentions
    }
    return tweet


# DATASET = 'geoengineering'
DATASET = 'climate'
BATCH_SIZE = 200000
SOURCE_FILE = f'data/{DATASET}/tweets_raw.jsonl'
TARGET_FILE = f'data/{DATASET}/tweets_clean.jsonl'


def get_batches(fp):
    while True:
        batch = [json.loads(next(fp)) for _ in range(BATCH_SIZE)]
        if len(batch) == 0:
            break
        yield batch


if __name__ == '__main__':
    if os.path.exists(TARGET_FILE):
        print(f'The file {TARGET_FILE} already exists. If you are sure you want to proceed, delete it first.')
        exit(1)

    with open(SOURCE_FILE, 'r') as f_in, open(TARGET_FILE, 'w') as f_out:
        for batch_i, tweets in enumerate(get_batches(f_in)):
            print(f'Processing batch {batch_i} ({batch_i * BATCH_SIZE:,} to {(batch_i + 1) * BATCH_SIZE:,}) ...')
            tweets = [process_tweet(t) for t in tweets]
            print('Writing batch...')
            [f_out.write(json.dumps(t) + '\n') for t in tweets]
