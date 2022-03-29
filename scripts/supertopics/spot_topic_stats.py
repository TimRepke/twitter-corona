from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist
from typing import Literal
import numpy as np
import json
from prettytable import PrettyTable

DATASET = 'climate2'
LIMIT = 7000000
DATE_FORMAT: DateFormat = 'monthly'
NORM: Literal['abs', 'col', 'row'] = 'abs'
BOOST = ['raw',  # 0
         'retweets',  # 1
         'replies',  # 2
         'likes',  # 3
         'retweets_likes',  # 4
         'replies_likes',  # 5
         'retweets_replies',  # 6
         'retweets_likes_replies'  # 7
         ][0]
# SOURCE_DIR = f'data/{DATASET}/topics_big2'
# TWEETS_FILE = f'data/{DATASET}/tweets_filtered_{LIMIT}.jsonl'
# LABELS_FILE = f'{SOURCE_DIR}/labels_{LIMIT}_tsne.npy'

FILE_SUPERTOPICS = f'data/{DATASET}/topics_big2/supertopics.csv'
FILE_TEMP_DIST = f'data/{DATASET}/topics_big2/temporal/{DATE_FORMAT}/temporal_{LIMIT}_{DATE_FORMAT}_{BOOST}_{NORM}.json'

groups, topics, distributions = read_temp_dist(FILE_TEMP_DIST)

annotations = read_supertopics(FILE_SUPERTOPICS)
spot_topics = get_spottopics(distributions, threshold=0.4, min_size=500)

# print(topics)
# print(distributions.sum(axis=0))
print(distributions.shape)
print(annotations.shape)
print(spot_topics.shape)
tab = PrettyTable(field_names=['supertopic', 'N topics', 'N spottopics', 'spots/topics',
                               'N tweets', 'N tweet spot', 'spottweets/tweets'])

for st in SuperTopic:
    n_topics = annotations[:, st].sum()
    n_spots = annotations[:, st][spot_topics].sum()
    n_topic_tweets = distributions.T[annotations[:, st] > 0].sum()
    n_spot_tweets = distributions.T[spot_topics][annotations[:, st][spot_topics] > 0].sum()
    tab.add_row([st,
                 f'{n_topics} ({n_topics / len(topics):.1%})',
                 f'{n_spots} ({n_spots / len(spot_topics):.1%})',
                 f'{n_spots / n_topics:.2%}',
                 f'{n_topic_tweets:,} ({n_topic_tweets / distributions.sum():.1%})',
                 f'{n_spot_tweets:,} ({n_spot_tweets / distributions.T[spot_topics].sum():.1%})',
                 f'{n_spot_tweets / n_topic_tweets:.1%}'
                 ])
    print(st, groups[distributions.T[spot_topics][annotations[:, st][spot_topics] > 0].sum(axis=0).argmax()])

print(tab)
print('annotated topics:', sum(annotations.sum(axis=1) > 0))
print('num topics:', len(topics))
print('num spot topics:', len(spot_topics))

# when does each spot topic "peak"
r = []
for spt in spot_topics:
    r.append((spt[0], groups[distributions.T[spt].argmax()]))
rs = sorted(r, key=lambda x: x[1])
print(rs)
