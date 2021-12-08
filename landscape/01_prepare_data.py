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


# https://github.com/dhs-gov/sentop/


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


def parse_tweet(t):
    txt = clean_tweet(t['text'], remove_hashtags=False, remove_urls=True, remove_nonals=True, remove_mentions=True)
    return ' '.join(txt.split(' ')[:256])


if __name__ == '__main__':
    LIMIT = 1000000
    CHUNK_SIZE = 1500
    INIT_SKIP = 0
    SOURCE_FILE = 'data/climate_tweets.jsonl'
    TARGET_FILE = 'data/climate_tweets_sentiment.jsonl'

    print('Loading tweets...')
    with open(SOURCE_FILE) as f:
        num_lines = sum(1 for l in f)
        print(f'  - Source file contains {num_lines} tweets.')

    SKIP_LINES = max(int(num_lines / LIMIT), 1)
    N_CHUNKS = math.ceil(num_lines / CHUNK_SIZE)
    print(f'  - Targeting to load {LIMIT} tweets by reading every {SKIP_LINES}th tweet...')

    with open(SOURCE_FILE) as f_in, open(TARGET_FILE, 'w') as f_out:
        line_num = 0
        for chunk in range(N_CHUNKS):
            print(f'===== PROCESSING CHUNK {chunk} ({(chunk + 1) * CHUNK_SIZE}/{num_lines}) =====')

            tweets = []
            while len(tweets) < CHUNK_SIZE and line_num < num_lines:
                if line_num > INIT_SKIP and (line_num % SKIP_LINES) == 0:
                    t = json.loads(next(f_in))
                    if t['text'] is not None and len(t['text']) > 5 and t['lang'] == 'en':
                        tweets.append(t)
                else:  # skip / ignore line
                    next(f_in)
                line_num += 1

            print(f'Current file pos: {line_num}; Tweets from {tweets[0]["created_at"]} to {tweets[-1]["created_at"]}')

            texts = [parse_tweet(t) for t in tweets]

            results = {}
            for model, info in MODELS.items():
                start = time.time()
                print(f'[{datetime.now()}] Applying model {model}...')
                results[model] = assess_tweets(texts, model=info['model'], labels=info['labels'])
                secs = time.time() - start
                print(f'  - Done after {secs // 60:.0f}m {secs % 60:.0f}s')

            for i, tweet in enumerate(tweets):
                tweet['sentiments'] = {m: results[m][i] for m in MODELS.keys()}
                f_out.write(json.dumps(tweet) + '\n')
