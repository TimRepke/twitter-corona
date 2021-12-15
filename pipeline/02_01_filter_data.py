from datetime import datetime
import json
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from utils.tweets import clean_tweet, read_tweets
import math
import time
import hashlib
import os

# DATASET = 'geoengineering'
DATASET = 'climate'
SOURCE_FILE = f'data/{DATASET}/tweets_clean.jsonl'

LIMIT = 10000
ONLY_EN = True
MIN_TOKENS = 4
MAX_HASHTAGS = 5

RELEVANCE_FILE = f'data/{DATASET}/tweets_relevant_{ONLY_EN}_{MIN_TOKENS}_{MAX_HASHTAGS}.txt'
IRRELEVANCE_FILE = f'data/{DATASET}/tweets_irrelevant_{ONLY_EN}_{MIN_TOKENS}_{MAX_HASHTAGS}.txt'
TARGET_FILE = f'data/{DATASET}/tweets_filtered_{LIMIT}.jsonl'


def is_relevant(tweet):
    if ONLY_EN and tweet['lang'] != 'en':
        return False
    if tweet['text'] is None:
        return False
    if tweet['meta']['n_tokens_raw'] <= MIN_TOKENS and tweet['meta']['n_hashtags'] >= MAX_HASHTAGS:
        return False
    return True


def get_hash(tweet):
    return hashlib.md5(f'{tweet["author_id"]}|{tweet["clean_text"].lower()}'.encode()).digest()


if __name__ == '__main__':
    if os.path.exists(TARGET_FILE):
        print(f'The file {TARGET_FILE} already exists. If you are sure you want to proceed, delete it first.')
        exit(1)

    if not os.path.exists(RELEVANCE_FILE):
        num_lines = 0
        n_duplicates = 0
        n_irrelevant = 0
        print('Filter and remove duplicates...')
        with open(SOURCE_FILE, 'r') as f_in, \
                open(RELEVANCE_FILE, 'w') as f_rel_out, \
                open(IRRELEVANCE_FILE, 'w') as f_irrel_out:
            hashes = set()
            for line_i, line in tqdm(enumerate(f_in)):
                tweet_o = json.loads(line)
                num_lines += 1

                is_en = (not ONLY_EN) or (ONLY_EN and tweet_o['lang'] != 'en')
                has_text = tweet_o['text'] is not None
                has_min_tokens = tweet_o['meta']['n_tokens_raw'] >= MIN_TOKENS
                has_max_hashtags = tweet_o['meta']['n_hashtags'] <= MAX_HASHTAGS

                # if is_relevant(tweet_o):
                if is_en and has_text and has_min_tokens and has_max_hashtags:
                    h = get_hash(tweet_o)
                    if h not in hashes:
                        # relevant and non-duplicate
                        f_rel_out.write(f'{line_i}\n')
                    else:
                        f_irrel_out.write(f'{line_i}|1|0|0|0|0\n')
                        n_duplicates += 1
                    hashes.add(h)
                else:
                    f_irrel_out.write(f'{line_i}|0|{is_en:d}|{has_text:d}|{has_min_tokens:d}|{has_max_hashtags:d}\n')
                    n_irrelevant += 1
            print(f'Read {num_lines:,} lines, found {n_duplicates:,} '
                  f'duplicates and {n_irrelevant:,} irrelevant tweets.')

            # clear up memory
            del hashes

        N_LINES = num_lines - n_duplicates - n_irrelevant
    else:
        print(f'I\'m using the already existing relevance file {RELEVANCE_FILE}!')
        with open(RELEVANCE_FILE) as f:
            N_LINES = sum(1 for _ in f)

    SKIP_LINES = max(int(N_LINES / LIMIT), 1)

    print(f'Aiming to reduce size of the dataset from {N_LINES:,} to {LIMIT:,} '
          f'by skipping {SKIP_LINES} relevant non-duplicate tweets.')
    with open(SOURCE_FILE) as f_source, open(RELEVANCE_FILE) as f_rel, open(TARGET_FILE, 'w') as f_out:
        line_source = 0
        line_rel = 0

        while line_rel < N_LINES:
            if (line_rel % SKIP_LINES) == 0:
                next_source_line = int(next(f_rel))
                line = None
                while line_source <= next_source_line:
                    line = next(f_source)
                    line_source += 1
                if line is None:
                    break
                f_out.write(line)
            else:  # skip / ignore line
                next(f_source)
            line_rel += 1
