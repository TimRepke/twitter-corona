import re
from datetime import datetime
from dataclasses import dataclass
import sqlite3
from collections import defaultdict
from typing import Optional
import numpy as np

from .embedding import BaseEmbedder, SentenceTransformerBackend

PATTERN_URL = re.compile(
    r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))')
PATTERN_HASHTAG = re.compile(r'#[a-zA-Z0-9_]+')
PATTERN_MENTION = re.compile(r'@[a-zA-Z0-9_]+')
PATTERN_NON_ALPH_NUM = re.compile(r'[^a-zA-Z0-9 ]')
PATTERN_MULTISPACE = re.compile(r'\s+')


@dataclass
class Tweet:
    id: int
    uid: int
    text: str
    clean_text: str
    date: datetime
    hashtags: list[str]
    mentions: list[str]
    urls: list[str]


def fetchall_dict(cur):
    header = [d[0] for d in cur.description]
    return [dict(zip(header, r)) for r in cur.fetchall()]


def remove_url(s):
    return PATTERN_URL.sub(' URL ', s)


def remove_nonal(s):
    return PATTERN_NON_ALPH_NUM.sub(' ', s)


def remove_hashtag(s):
    return PATTERN_HASHTAG.sub(' HASHTAG ', s)


def remove_mention(s):
    return PATTERN_MENTION.sub(' MENTION ', s)


def get_hashtags(s):
    return PATTERN_HASHTAG.findall(s)


def get_mentions(s):
    return PATTERN_MENTION.findall(s)


def get_urls(s):
    return PATTERN_URL.findall(s)


def clean_tweet(s, remove_hashtags=True, remove_urls=True, remove_mentions=True, remove_nonals=True):
    if remove_urls:
        s = remove_url(s)
    if remove_hashtags:
        s = remove_hashtag(s)
    if remove_mentions:
        s = remove_mention(s)
    if remove_nonals:
        s = remove_nonal(s)
    return PATTERN_MULTISPACE.sub(' ', s)


def s2time(s):
    return datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S')


def process_tweet(t: dict, remove_hashtags=True, remove_urls=True,
                  remove_mentions=True, remove_nonals=True) -> Tweet:
    s = t['text']

    return Tweet(
        id=t['id'],
        uid=t['author_id'],
        date=s2time(t['created_at']),
        text=s,
        clean_text=clean_tweet(s, remove_hashtags, remove_urls, remove_mentions, remove_nonals),
        hashtags=get_hashtags(s),
        mentions=get_mentions(s),
        urls=get_urls(s)
    )


class Tweets:
    def __init__(self, db_file, remove_hashtags=True, remove_urls=True, remove_mentions=True, remove_nonals=True,
                 limit: int = None):
        self.limit = f'LIMIT {limit}' if limit is not None else ''

        connection = sqlite3.connect(db_file)
        self.cursor = connection.cursor()

        print('Loading tweets...')
        self.cursor.execute(f'SELECT * FROM pooled_sample_tweets {self.limit};')
        tweets_d = fetchall_dict(self.cursor)
        self.tweets = [process_tweet(tweet, remove_hashtags, remove_urls, remove_mentions, remove_nonals)
                       for tweet in tweets_d]

    def __len__(self):
        return len(self.tweets)

    def groupby_date(self, fmt) -> dict[list]:
        ret = defaultdict(list)
        [ret[li.date.strftime(fmt)].append(li) for li in self.tweets]
        return ret

    def get_embeddings(self, model: Optional[BaseEmbedder] = None) -> np.ndarray:
        if model is None:
            model = SentenceTransformerBackend("paraphrase-multilingual-MiniLM-L12-v2")

        return model.embed_documents([t.clean_text for t in self.tweets],
                                     verbose=True)

    def histogram(self, fmt) -> list[dict]:
        self.cursor.execute(
            f"""
                SELECT strftime('{fmt}', created_at) grp, count(1) freq
                FROM (SELECT * FROM pooled_sample_tweets {self.limit})
                GROUP BY grp
                ORDER BY grp;
            """)
        return fetchall_dict(self.cursor)
