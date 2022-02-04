from datetime import datetime
import json
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from utils.tweets import clean_tweet, read_tweets, s2time
import math
import time
from typing import Literal
import plotly.graph_objects as go
from colorcet import glasbey
import hashlib
import os
from collections import defaultdict
from utils.tweets import get_hash

# DATASET = 'geoengineering'
DATASET = 'climate2'
SOURCE_FILE = f'data/{DATASET}/tweets_clean.jsonl'

TARGET_FOLDER = f'data/{DATASET}/stats'
os.makedirs(TARGET_FOLDER, exist_ok=True)

MIN_TOKENS = 4
MAX_HASHTAGS = 5
DATE_FORMAT: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'

FORMATS = {'yearly': '%Y', 'monthly': '%Y-%m', 'weekly': '%Y-%W', 'daily': '%Y-%m-%d'}
FORMAT = FORMATS[DATE_FORMAT]

if __name__ == '__main__':
    groups = defaultdict(lambda: {
        'total': 0,
        'duplicate': 0,
        'not_en': 0,
        'lang_null': 0,
        'leq_min_tokens': 0,
        'geq_max_hashtags': 0,
        'relevant': 0,
        'not_relevant': 0
    })
    print('Processing...')
    with open(SOURCE_FILE, 'r') as f_in:
        hashes = set()
        for line_i, line in tqdm(enumerate(f_in)):
            tweet_o = json.loads(line)
            lang = tweet_o.get('lang', None)
            is_en = lang == 'en'
            is_en_or_null = is_en or lang is None
            has_min_tokens = tweet_o['meta']['n_tokens_raw'] >= MIN_TOKENS
            has_max_hashtags = tweet_o['meta']['n_hashtags'] <= MAX_HASHTAGS

            grp = s2time(tweet_o['created_at']).strftime(FORMAT)

            groups[grp]['total'] += 1
            if is_en_or_null and has_min_tokens and has_max_hashtags:
                h = get_hash(tweet_o, include_author=False)
                if h not in hashes:
                    # relevant and non-duplicate
                    groups[grp]['relevant'] += 1
                else:
                    groups[grp]['not_relevant'] += 1
                    groups[grp]['duplicate'] += 1
                hashes.add(h)
            else:
                groups[grp]['not_relevant'] += 1
                if not is_en:
                    groups[grp]['not_en'] += 1
                    if lang is None:
                        groups[grp]['lang_null'] += 1
                if not has_min_tokens:
                    groups[grp]['leq_min_tokens'] += 1
                if not has_max_hashtags:
                    groups[grp]['geq_max_hashtags'] += 1

        # clear up memory
        del hashes

    srt_grps = list(sorted(groups.keys()))
    axis = [f'd:{k}' for k in srt_grps]
    for key in ['total', 'duplicate', 'not_en', 'lang_null',
                'leq_min_tokens', 'geq_max_hashtags', 'relevant', 'not_relevant']:
        values = [groups[k][key] for k in srt_grps]

        fig = go.Figure([go.Bar(x=axis,
                                y=values)])
        fig.write_html(f'{TARGET_FOLDER}/histogram_{DATE_FORMAT}_{key}.html')
