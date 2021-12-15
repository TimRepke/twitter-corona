from adaptnlp import EasySequenceClassifier

from datetime import datetime
# from senti import preprocess
import json
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from utils.tweets import clean_tweet, read_tweets
import math
import time
import os

# https://github.com/dhs-gov/sentop/


MODELS = {
    # to find more models, browse this page:
    # https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
    # Hint: the search function doesn't really work...
    'cardiff-sentiment': {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/sentiment/mapping.txt
        'model': 'cardiffnlp/twitter-roberta-base-sentiment',
        'labels': ['negative', 'neutral', 'positive']
    },
    'cardiff-emotion': {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/emotion/mapping.txt
        'model': 'cardiffnlp/twitter-roberta-base-emotion',
        'labels': ['anger', 'joy', 'optimism', 'sadness']
    },
    'cardiff-offensive': {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/offensive/mapping.txt
        'model': 'cardiffnlp/twitter-roberta-base-offensive',
        'labels': ['not-offensive', 'offensive']
    },
    'cardiff-stance-climate': {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/stance/mapping.txt
        'model': 'cardiffnlp/twitter-roberta-base-stance-climate',
        'labels': ['none', 'against', 'favor']
    },
    'geomotions-orig': {
        # https://huggingface.co/monologg/bert-base-cased-goemotions-original/blob/main/config.json
        'model': 'monologg/bert-base-cased-goemotions-original',
        'labels': [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
            'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise',
        ]
    },
    'geomotions-ekman': {
        # https://huggingface.co/monologg/bert-base-cased-goemotions-ekman/blob/main/config.json
        'model': 'monologg/bert-base-cased-goemotions-ekman',
        'labels': ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    },
    # 'nlptown-sentiment': {
    #     # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/blob/main/config.json
    #     'model': 'nlptown/bert-base-multilingual-uncased-sentiment',
    #     'labels': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    # },
    'bertweet-sentiment': {
        # https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
        'model': 'finiteautomata/bertweet-base-sentiment-analysis',
        'labels': ['negative', 'neutral', 'positive']
    },
    'bertweet-emotions': {
        # https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis
        'model': 'finiteautomata/bertweet-base-emotion-analysis',
        'labels': ['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear']
    },
    # 'bert-sst2': {
    #     # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/blob/main/config.json
    #     'model': 'distilbert-base-uncased-finetuned-sst-2-english',
    #     'labels': ['negative', 'positive']
    # }
}


def produce_batches(fp, batch_size, init_skip=0):
    print('Counting tweets...')
    num_lines = sum(1 for l in fp)
    print(f'  - Source file contains {num_lines} tweets.')
    n_batches = math.ceil(num_lines / batch_size)
    fp.seek(0)

    line_num = 0
    for _ in range(init_skip):
        next(f_in)
        line_num += 1

    for batch_i in range(n_batches):
        print(f'===== PROCESSING BATCH {batch_i + 1} ({(batch_i + 1) * BATCH_SIZE}/{num_lines}) =====')

        tweets = []
        while len(tweets) < BATCH_SIZE and line_num < num_lines:
            tweets.append(json.loads(next(f_in)))
            line_num += 1

        print(f'Current file pos: {line_num}; Tweets from {tweets[0]["created_at"]} to {tweets[-1]["created_at"]}')
        yield tweets


def assess_tweets(texts: list[str], model, labels):
    def process_result(scores):
        srt = np.argsort(scores)
        # return [f'{labels[i]} ({scores[i]:.2f})' for i in reversed(srt)]
        scores_lst = scores.tolist()
        return [(labels[i], scores_lst[i]) for i in reversed(srt)]

    classifier = EasySequenceClassifier()
    res = classifier.tag_text(
        text=texts,
        model_name_or_path=model
    )
    return [
        process_result(r)
        for r in res['probs']
    ]


if __name__ == '__main__':
    # DATASET = 'geoengineering'
    DATASET = 'climate'

    SKIP_FIRST_N_LINES = 0  # can be used to continue a run that failed
    BATCH_SIZE = 1500
    LIMIT = 10000

    SOURCE_FILE = f'data/{DATASET}/tweets_filtered_{LIMIT}.jsonl'
    TARGET_FILE = f'data/{DATASET}/tweets_sentiment_{LIMIT}.jsonl'

    if os.path.exists(TARGET_FILE):
        print(f'The file {TARGET_FILE} already exists. If you are sure you want to proceed, delete it first.')
        exit(1)

    with open(SOURCE_FILE) as f_in, open(TARGET_FILE, 'w') as f_out:
        for tweets_batch in produce_batches(f_in, BATCH_SIZE, SKIP_FIRST_N_LINES):
            texts = [t['clean_text'] for t in tweets_batch]

            results = {}
            for model, info in MODELS.items():
                start = time.time()
                print(f'[{datetime.now()}] Applying model {model}...')
                results[model] = assess_tweets(texts, model=info['model'], labels=info['labels'])
                secs = time.time() - start
                print(f'  - Done after {secs // 60:.0f}m {secs % 60:.0f}s')

            for i, tweet in enumerate(tweets_batch):
                tweet['sentiments'] = {m: results[m][i] for m in MODELS.keys()}
                f_out.write(json.dumps(tweet) + '\n')
