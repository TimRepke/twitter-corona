import re
import json
from datetime import datetime
from dataclasses import dataclass
import sqlite3
from collections import defaultdict
from typing import Optional, List, Tuple
import numpy as np

from .embedding import BaseEmbedder, SentenceTransformerBackend

# PATTERN_URL = re.compile(r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))')
PATTERN_URL = re.compile(r'(https?://[^\s]+)')
PATTERN_HASHTAG = re.compile(r'#[a-zA-Z0-9_]+')
PATTERN_MENTION = re.compile(r'@[a-zA-Z0-9_]+')
PATTERN_NON_ALPH_NUM = re.compile(r'[^a-zA-Z0-9 ]')
PATTERN_NON_ALPH = re.compile(r'[^a-zA-Z ]')
PATTERN_MULTISPACE = re.compile(r'\s+')


@dataclass
class Tweet:
    id: int
    uid: int
    text: str
    clean_text: str
    date: datetime
    hashtags: List[str]
    mentions: List[str]
    urls: List[str]


def fetchall_dict(cur):
    header = [d[0] for d in cur.description]
    return [dict(zip(header, r)) for r in cur.fetchall()]


def remove_url(s):
    # return PATTERN_URL.sub(' URL ', s)
    return PATTERN_URL.sub(' ', s)


def remove_nonalnum(s):
    return PATTERN_NON_ALPH_NUM.sub(' ', s)


def remove_nonal(s):
    return PATTERN_NON_ALPH.sub(' ', s)


def remove_hashtag(s):
    # return PATTERN_HASHTAG.sub(' HASHTAG ', s)
    return PATTERN_HASHTAG.sub(' ', s)


def remove_mention(s):
    # return PATTERN_MENTION.sub(' MENTION ', s)
    return PATTERN_MENTION.sub(' ', s)


def get_hashtags(s):
    return PATTERN_HASHTAG.findall(s)


def get_mentions(s):
    return PATTERN_MENTION.findall(s)


def get_urls(s):
    return PATTERN_URL.findall(s)


def clean_tweet(s, remove_hashtags=True, remove_urls=True, remove_mentions=True, remove_nonals=True,
                remove_nonalnums=False):
    if s is None:
        return s
    if remove_urls:
        s = remove_url(s)
    if remove_hashtags:
        s = remove_hashtag(s)
    if remove_mentions:
        s = remove_mention(s)
    if remove_nonalnums:
        s = remove_nonalnum(s)
    if remove_nonals:
        s = remove_nonal(s)
    return PATTERN_MULTISPACE.sub(' ', s).strip()


def s2time(s):
    s = s.replace('T', ' ')
    return datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S')


def process_tweet(t: dict, remove_hashtags=True, remove_urls=True,
                  remove_mentions=True, remove_nonals=True) -> Tweet:
    s = t['text']

    return Tweet(
        id=t.get('id', 1),
        uid=t.get('author_id', 1),
        date=s2time(t['created_at']),
        text=s,
        clean_text=clean_tweet(s, remove_hashtags, remove_urls, remove_mentions, remove_nonals),
        hashtags=get_hashtags(s),
        mentions=get_mentions(s),
        urls=get_urls(s)
    )


def read_tweets(filename, limit):
    with open(filename) as f:
        num_lines = sum(1 for _ in f)
    skip_lines = max(int(num_lines / limit), 1)

    with open(filename) as f:
        tweets_d = [json.loads(l) for i, l in enumerate(f) if (i % skip_lines) == 0]
        tweets_d = [t for t in tweets_d if t['text'] is not None and len(t['text']) > 5 and t['lang'] == 'en']
        return tweets_d


class Tweets:
    def __init__(self, db_file=None, filename=None,
                 remove_hashtags=True, remove_urls=True, remove_mentions=True, remove_nonals=True,
                 limit: int = None):

        assert db_file or filename

        if db_file is not None:
            raise NotImplementedError('This was removed!')
        else:
            tweets_d = read_tweets(filename, limit)

        self.tweets = [process_tweet(tweet, remove_hashtags, remove_urls, remove_mentions, remove_nonals)
                       for tweet in tweets_d]

    def __len__(self):
        return len(self.tweets)

    def groupby_date(self, fmt) -> "dict[list]":
        ret = defaultdict(list)
        [ret[li.date.strftime(fmt)].append(li) for li in self.tweets]
        return ret

    def get_embeddings(self, model: Optional[BaseEmbedder] = None) -> np.ndarray:
        if model is None:
            model = SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")

        return model.embed_documents([t.clean_text for t in self.tweets],
                                     verbose=True)

    def histogram(self, fmt) -> List[dict]:
        self.cursor.execute(
            f"""
                SELECT strftime('{fmt}', created_at) grp, count(1) freq
                FROM (SELECT * FROM pooled_sample_tweets {self.limit})
                GROUP BY grp
                ORDER BY grp;
            """)
        return fetchall_dict(self.cursor)
