import numpy as np
from tqdm import tqdm
import json

FILE_TSNE = 'data/geoengineering/layout_tsne.npy'
FILE_TWEETS = 'data/geoengineering/tweets_classified2.jsonl'
FILE_OUT = 'data/geoengineering/landscape.json'

LABEL_MAP = {
    'cardiff-emotion': ['anger', 'joy', 'optimism', 'sadness'],
    'cardiff-sentiment': ['negative', 'neutral', 'positive'],
    'cardiff-offensive': ['not-offensive', 'offensive'],
    'geomotions-ekman': ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'],
    'bertweet-emotions': ['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear'],
    'nrc': ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'positive',
            'negative', 'sadness', 'disgust', 'joy']
}

print('Reading tsne...')
with open(FILE_TSNE, 'rb') as f:
    TSNE_FULL = np.load(f)
EPS = 1e-12


def produce(tweet, p):
    return [
        int(tweet['id']),
        tweet['sid'],
        # tweet['created_at'][:19],
        p[0], p[1],
        LABEL_MAP['cardiff-emotion'].index(list(tweet['classes']['cardiff-emotion'].keys())[0]),
        LABEL_MAP['cardiff-sentiment'].index(list(tweet['classes']['cardiff-sentiment'].keys())[0]),
        LABEL_MAP['cardiff-offensive'].index(list(tweet['classes']['cardiff-offensive'].keys())[0]),
        LABEL_MAP['geomotions-ekman'].index(list(tweet['classes']['geomotions-ekman'].keys())[0]),
        LABEL_MAP['bertweet-emotions'].index(list(tweet['classes']['bertweet-emotions'].keys())[0]),
        [LABEL_MAP['nrc'].index(la) for la in list(tweet['classes']['nrc'].keys())]
    ]


with open(FILE_TWEETS, 'r') as f_in:
    tweets = [
        produce(json.loads(line), pos)
        for line, pos in tqdm(zip(f_in, TSNE_FULL))]

with open(FILE_OUT, 'w') as f_out:
    json.dump(tweets, f_out)
