from collections import defaultdict
import numpy as np
import json

from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)

TARGET_FILE = 'data/geoengineering/topics_finn3/full_dump.json'
FILE_TWEETS = 'data/geoengineering/tweets_classified2.jsonl'
FILE_TSNE_FILTERED = 'data/geoengineering/layout_tsne_filtered.npy'
FILE_TOPICS = 'data/geoengineering/topics_finn3.npy'
FILTER = {'c_01', 'c_02', 'c_03', 'c_04', 'c_05', 'c_06', 'c_07', 'c_08', 'c_55', 'c_09'}
START = (2007, 1)
END = (2021, 12)
N_TOPIC_TOKENS = 20

logging.info('Reading TSNE data...')
with open(FILE_TSNE_FILTERED, 'rb') as f:
    TSNE = np.load(f)

logging.info('Reading topic annotations...')
with open(FILE_TOPICS, 'rb') as f:
    TOPICS = np.load(f)

logging.info(f'Generation groups for time from {START} to {END}')
groups = [
    f'{yr}-{mnth:02n}'
    for yr in range(START[0], END[0] + 1)
    for mnth in range(1, 12 + 1)
    if not (yr == START[0] and mnth < START[1]) and not (yr == END[0] and mnth > END[1])
]
groups_map = {g: i for i, g in enumerate(groups)}
num_topics = np.unique(TOPICS).shape[0]
logging.debug(f'Created {len(groups)} groups and found {num_topics} topics.')

logging.info('Prepping counters...')
topic_tweets = defaultdict(list)
topic_counts_raw = np.zeros((num_topics, len(groups)))
topic_counts_likes = np.zeros((num_topics, len(groups)))
topic_counts_rt = np.zeros((num_topics, len(groups)))
topic_counts_quote = np.zeros((num_topics, len(groups)))
topic_counts_repl = np.zeros((num_topics, len(groups)))

logging.info('Reading and processing tweets...')
tweets = []
with open(FILE_TWEETS) as f:
    i = 0

    for line in tqdm(f):
        tweet = json.loads(line)
        if tweet['sid'][0] in FILTER:
            topic = TOPICS[i]
            tsne = TSNE[i]
            group = tweet['created_at'][:7]
            group_i = groups_map[group]

            topic_counts_raw[topic][group_i] += 1
            topic_counts_likes[topic][group_i] += tweet['public_metrics']['like_count']
            topic_counts_rt[topic][group_i] += tweet['public_metrics']['retweet_count']
            topic_counts_repl[topic][group_i] += tweet['public_metrics']['reply_count']
            topic_counts_quote[topic][group_i] += tweet['public_metrics']['quote_count']

            topic_tweets[topic].append(tweet['clean_text'])

            tweets.append({
                'time': tweet['created_at'],
                'id': tweet['id'],
                'group': group,
                'topic': int(topic),
                'text': tweet['text'],
                'retweets': tweet['public_metrics']['retweet_count'],
                'likes': tweet['public_metrics']['like_count'],
                'replies': tweet['public_metrics']['reply_count']
            })
            i += 1
logging.debug(f'Included {len(tweets)} tweets after filtering.')

logging.info('Constructing tf-idf vectors...')
stop_words = list(ENGLISH_STOP_WORDS) + ['url', 'mention', 'hashtag', 'rt']
vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=1.0, min_df=1, ngram_range=(1, 2), max_features=None,
                             lowercase=True, use_idf=True, smooth_idf=True)
vectors = vectorizer.fit_transform([' '.join(topic_tweets[tk]) for tk in sorted(topic_tweets.keys())])
vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
logging.debug(f'Total vocabulary: {len(vectorizer.vocabulary_)} unique tokens')

logging.info('Constructing topic list...')
topics = []
for topic_i in sorted(topic_tweets.keys()):
    token_scores = np.argsort(vectors[topic_i].todense())
    top_tokens = [
        (vocab[token_scores[0, -(token_i + 1)]], vectors[topic_i, token_scores[0, -(token_i + 1)]])
        for token_i in range(N_TOPIC_TOKENS)
    ]
    topics.append({
        'tfidf': ', '.join([t[0] for t in top_tokens]),
        'mmr': 'NOT COMPUTED',
        'n_tweets': len(topic_tweets[topic_i]),
        'abs_raw': topic_counts_raw[topic_i].tolist(),
        'abs_likes': topic_counts_likes[topic_i].tolist(),
        'abs_replies': topic_counts_repl[topic_i].tolist(),
        'abs_retweets': topic_counts_rt[topic_i].tolist()
    })

logging.info('Writing dump...')
with open(TARGET_FILE, 'w') as f_out:
    json.dump({
        'groups': groups,
        'tweets': tweets,
        'topics': topics
    }, f_out)

logging.info('All done!')
