import numpy as np
from tqdm import tqdm
import json
from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
from matplotlib import pyplot as plt
from itertools import chain, repeat
import pandas as pd
from sklearn.neighbors import KernelDensity
from multiprocessing import Pool
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
MODEL = 'geomotions-ekman'
print('Reading tsne...')
with open(FILE_TSNE, 'rb') as f:
    TSNE_FULL = np.load(f)
lmap = {k: i for i, k in enumerate(LABEL_MAP[MODEL], start=1)}
lmap['NONE'] = 0


def get_label(t):
    c = t['classes'][MODEL]
    if len(c) == 0:
        return 0
    if len(c) == 1:
        return lmap[list(c.keys())[0]]
    return lmap[sorted(list(c.items()), key=lambda x: x[1])[-1][0]]


with open(FILE_TWEETS) as f:
    labels = np.array([get_label(json.loads(line)) for line in tqdm(f)])

plt.figure(figsize=(15, 15), dpi=150)
for q, i in lmap.items():
    if i == 5 or i==0:
        continue
    plt.scatter(TSNE_FULL[labels == i][:, 0], TSNE_FULL[labels == i][:, 1], marker='X', alpha=0.1, s=0.05, label=q)
    print(f'{q}: {len(TSNE_FULL[labels == i]):,} tweets')

plt.ylim(-25, 25)
plt.xlim(-30, 30)
plt.tight_layout()
plt.title(MODEL)
plt.legend()
plt.show()
