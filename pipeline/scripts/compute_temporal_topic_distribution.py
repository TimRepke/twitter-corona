from utils.topics.utils import date2group
import json
import numpy as np
from typing import Literal
import os
from tqdm import tqdm

DATASET = 'climate2'
LIMIT = 7000000
DATE_FORMAT: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'daily'
EPS = 1e-12

SOURCE_DIR = f'data/{DATASET}/topics_big2'
TWEETS_FILE = f'data/{DATASET}/tweets_filtered_{LIMIT}.jsonl'
LABELS_FILE = f'{SOURCE_DIR}/labels_{LIMIT}_tsne.npy'

TARGET_DIR = f'{SOURCE_DIR}/temporal/{DATE_FORMAT}'
os.makedirs(TARGET_DIR, exist_ok=True)

print('Loading labels...')
labels = np.load(LABELS_FILE)
topic_ids = np.unique(labels, return_counts=False)


def tweets():
    with open(TWEETS_FILE) as f_tweets:
        for line in f_tweets:
            yield json.loads(line)


print('Reading and counting tweets per topic per time group...')
groups = {}
for tweet, topic in tqdm(zip(tweets(), labels)):
    group = date2group(tweet['created_at'], DATE_FORMAT)

    if group not in groups:
        groups[group] = {
            'raw': np.zeros_like(topic_ids),
            'retweets': np.zeros_like(topic_ids),
            'likes': np.zeros_like(topic_ids),
            'replies': np.zeros_like(topic_ids)
        }

    groups[group]['raw'][topic] += 1
    groups[group]['retweets'][topic] += tweet.get('retweets_count', 0)
    groups[group]['likes'][topic] += tweet.get('likes_count', 0)
    groups[group]['replies'][topic] += tweet.get('replies_count', 0)

print('Rearranging counts into np arrays...')
time_groups = sorted(groups.keys())
distributions = {
    'raw': np.array([groups[group]['raw'].tolist() for group in time_groups]),
    'retweets': np.array([groups[group]['retweets'].tolist() for group in time_groups]),
    'likes': np.array([groups[group]['likes'].tolist() for group in time_groups]),
    'replies': np.array([groups[group]['replies'].tolist() for group in time_groups])
}

for boost in [[], ['retweets'], ['replies'], ['likes'], ['retweets', 'likes'], ['replies', 'likes'],
              ['retweets', 'replies'], ['retweets', 'likes', 'replies']]:
    boost_prefix = '_'.join(boost or ['raw'])

    distribution = distributions['raw']
    for b in boost:
        distribution += distributions[b]

    for norm in ['col', 'row', 'abs']:
        print(f'Computing temporal distribution for boost: "{boost_prefix}" and normalisation: "{norm}"')

        if norm == 'row':
            topic_dist = distribution / (distribution.sum(axis=0) + EPS)
        elif norm == 'col':
            topic_dist = (distribution.T / (distribution.sum(axis=1) + EPS)).T
        else:  # norm == 'abs':
            topic_dist = distribution.copy()

        with open(f'{TARGET_DIR}/temporal_{LIMIT}_{DATE_FORMAT}_{boost_prefix}_{norm}.json', 'w') as f_out:
            f_out.write(json.dumps({
                'groups': time_groups,
                'topics': topic_ids.tolist(),
                'distribution': topic_dist.tolist()
            }))
