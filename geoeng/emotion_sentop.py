from adaptnlp import EasySequenceClassifier

# from senti import preprocess
import json
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from utils.tweets import clean_tweet
import math


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
    'nlptown-sentiment': {
        # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/blob/main/config.json
        'model': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'labels': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    },
    'bertweet-sentiment': {
        # https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
        'model': 'finiteautomata/bertweet-base-sentiment-analysis',
        'labels': ['negative', 'neutral', 'positive']
    },
    'bert-sst2': {
        # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/blob/main/config.json
        'model': 'distilbert-base-uncased-finetuned-sst-2-english',
        'labels': ['negative', 'positive']
    }
}


def parse_tweet(t):
    txt = clean_tweet(t['text'], remove_hashtags=False, remove_urls=True, remove_nonals=True, remove_mentions=True)

    return ' '.join(txt.split(' ')[:250])


if __name__ == '__main__':
    TOTAL = 1642400
    CHUNK_SIZE = 1000
    N_CHUNKS = math.ceil(TOTAL / CHUNK_SIZE)
    with open('data/geoengineering_tweets_tweets.jsonl') as f_in, \
            open('data/geoengineering_tweets_sentop.jsonl', 'a') as f_out:
        for chunk in range(N_CHUNKS):
            print(f'===== PROCESSING CHUNK {chunk} ({(chunk + 1) * CHUNK_SIZE}/{TOTAL}) =====')
            tweets = [json.loads(next(f_in)) for _ in range(CHUNK_SIZE)]
            tweets = [t for t in tweets if t['text'] is not None and len(t['text']) > 5]
            texts = [parse_tweet(t) for t in tweets]

            results = {}
            for model, info in MODELS.items():
                results[model] = assess_tweets(texts, model=info['model'], labels=info['labels'])

            for i, tweet in enumerate(tweets):
                tweet['sentiments'] = {m: results[m][i] for m in MODELS.keys()}
                f_out.write(json.dumps(tweet) + '\n')
