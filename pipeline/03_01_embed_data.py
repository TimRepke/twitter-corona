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
from utils.embedding import SentenceTransformerBackend


def line2txt_hashtags(line):
    tweet = json.loads(line)
    return tweet['clean_text'] + (' '.join(tweet['meta']['hashtags']))


def line2txt_clean(line):
    tweet = json.loads(line)
    return tweet['clean_text']


if __name__ == '__main__':
    # DATASET = 'geoengineering'
    DATASET = 'climate'

    # EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
    EMBEDDING_MODEL = 'vinai/bertweet-large'

    LIMIT = 10000
    INCLUDE_HASHTAGS = True
    SOURCE_FILE = f'data/{DATASET}/tweets_filtered_{LIMIT}.jsonl'
    TARGET_FILE = f'data/{DATASET}/tweets_embeddings_{LIMIT}_{INCLUDE_HASHTAGS}_' \
                  f'{EMBEDDING_MODEL.replace("/", "_")}.npy'

    if os.path.exists(TARGET_FILE):
        print(f'The file {TARGET_FILE} already exists. If you are sure you want to proceed, delete it first.')
        exit(1)

    print('Loading texts...')
    with open(SOURCE_FILE) as f_in:
        if INCLUDE_HASHTAGS:
            texts = [line2txt_hashtags(l) for l in f_in]
        else:
            texts = [line2txt_clean(l) for l in f_in]

    print('Embedding texts...')
    model = SentenceTransformerBackend(EMBEDDING_MODEL)
    embeddings = model.embed_documents(texts, verbose=True)

    print('Storing embeddings...')
    np.save(TARGET_FILE, embeddings)
