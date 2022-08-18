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
MODEL = 'nrc'
print('Reading tsne...')
with open(FILE_TSNE, 'rb') as f:
    TSNE_FULL = np.load(f)

for label in LABEL_MAP[MODEL]:
    print(label)
    with open(FILE_TWEETS) as f:
        bin_labels = np.array([int(json.loads(line)['classes'][MODEL].get(label, 0) > 0) for line in tqdm(f)])

    plt.figure(figsize=(15, 15), dpi=150)
    plt.scatter(TSNE_FULL[bin_labels == 0][:, 0], TSNE_FULL[bin_labels == 0][:, 1], marker='X', alpha=0.1, s=0.1)
    plt.scatter(TSNE_FULL[bin_labels == 1][:, 0], TSNE_FULL[bin_labels == 1][:, 1], marker='X', alpha=0.1, s=0.1)
    plt.title(label)
    plt.ylim(-25, 25)
    plt.xlim(-30, 30)
    plt.show()
