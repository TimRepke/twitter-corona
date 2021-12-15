import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from adaptnlp import EasySequenceClassifier
from tqdm import tqdm
from utils.io import exit_if_exists, produce_batches

# https://github.com/dhs-gov/sentop/


def assess_tweets(texts: "list[str]", model, labels):
    def process_result(scores):
        srt = np.argsort(scores)
        # return [f'{labels[i]} ({scores[i]:.2f})' for i in reversed(srt)]
        scores_lst = scores.tolist()
        return [(labels[i], scores_lst[i]) for i in reversed(srt)]

    classifier = EasySequenceClassifier()
    res = classifier.tag_text(text=texts, model_name_or_path=model)
    return [process_result(r) for r in res["probs"]]


MODELS = {
    # to find more models, browse this page:
    # https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
    # Hint: the search function doesn't really work...
    "cardiff-sentiment": {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/sentiment/mapping.txt
        "model": "cardiffnlp/twitter-roberta-base-sentiment",
        "labels": ["negative", "neutral", "positive"],
    },
    "cardiff-emotion": {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/emotion/mapping.txt
        "model": "cardiffnlp/twitter-roberta-base-emotion",
        "labels": ["anger", "joy", "optimism", "sadness"],
    },
    "cardiff-offensive": {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/offensive/mapping.txt
        "model": "cardiffnlp/twitter-roberta-base-offensive",
        "labels": ["not-offensive", "offensive"],
    },
    "cardiff-stance-climate": {
        # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/stance/mapping.txt
        "model": "cardiffnlp/twitter-roberta-base-stance-climate",
        "labels": ["none", "against", "favor"],
    },
    "geomotions-orig": {
        # https://huggingface.co/monologg/bert-base-cased-goemotions-original/blob/main/config.json
        "model": "monologg/bert-base-cased-goemotions-original",
        "labels": [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "neutral",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
        ],
    },
    "geomotions-ekman": {
        # https://huggingface.co/monologg/bert-base-cased-goemotions-ekman/blob/main/config.json
        "model": "monologg/bert-base-cased-goemotions-ekman",
        "labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    },
    # 'nlptown-sentiment': {
    #     # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/blob/main/config.json
    #     'model': 'nlptown/bert-base-multilingual-uncased-sentiment',
    #     'labels': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    # },
    "bertweet-sentiment": {
        # https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
        "model": "finiteautomata/bertweet-base-sentiment-analysis",
        "labels": ["negative", "neutral", "positive"],
    },
    "bertweet-emotions": {
        # https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis
        "model": "finiteautomata/bertweet-base-emotion-analysis",
        "labels": ["others", "joy", "sadness", "anger", "surprise", "disgust", "fear"],
    },
    # 'bert-sst2': {
    #     # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/blob/main/config.json
    #     'model': 'distilbert-base-uncased-finetuned-sst-2-english',
    #     'labels': ['negative', 'positive']
    # }
}


def classify_tweets(
    dataset: str,
    limit: int,
    skip_first_n_lines: int,
    batch_size: bool,
    source_f: Optional[str] = None,
    target_f: Optional[str] = None,
    models: Optional[Dict[str, Any]] = None
):

    if source_f is None:
        source_f = f"data/{dataset}/tweets_filtered_{limit}.jsonl"
    if target_f is None:
        target_f = f"data/{dataset}/tweets_sentiment_{limit}.jsonl"
    if models is None:
        models = MODELS

    exit_if_exists(target_f)

    with open(source_f) as f_in, open(target_f, "w") as f_out:
        for tweets_batch in tqdm(produce_batches(f_in, batch_size, skip_first_n_lines)):
            texts = [t["clean_text"] for t in tweets_batch]

            results = {}
            for model_name, info in models.items():
                start = time.time()
                tqdm.write(f"[{datetime.now()}] Applying model {model_name}...")
                results[model_name] = assess_tweets(
                    texts, model=info["model"], labels=info["labels"]
                )
                secs = time.time() - start
                tqdm.write(f"  - Done after {secs // 60:.0f}m {secs % 60:.0f}s")

            for i, tweet in enumerate(tweets_batch):
                tweet["sentiments"] = {m: results[m][i] for m in MODELS.keys()}
                f_out.write(json.dumps(tweet) + "\n")


if __name__ == "__main__":

    classify_tweets(
        dataset="climate",  # 'geoengineering'
        limit=10000,
        skip_first_n_lines=0,
        batch_size=1500,
    )