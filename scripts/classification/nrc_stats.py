import json
import re
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

FILE_TWEETS = 'data/geoengineering/tweets_classified2.jsonl'
LABELS = ['fear', 'anger', 'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust', 'joy', 'anticipation']

with open(FILE_TWEETS, 'r') as f_in:
    nrc_labels = [json.loads(line)['classes']['nrc'] for line in tqdm(f_in)]

# unwrap scores
scores = {lab: [] for lab in LABELS}
for labs in nrc_labels:
    for lab in LABELS:
        scores[lab].append(labs.get(lab, .0))
scores = {lab: np.array(s) for lab, s in scores.items()}

# plot distribution of values (all)
fig = plt.figure(figsize=(10, 20))
for i, lab in enumerate(LABELS, start=1):
    ax = plt.subplot(len(LABELS), 1, i)
    ax.hist(scores[lab], bins=50, range=(0, 1))
    ax.set_title(lab)
plt.tight_layout()
plt.show()

# plot distribution of values (only non-zero)
fig = plt.figure(figsize=(10, 20))
for i, lab in enumerate(LABELS, start=1):
    ax = plt.subplot(len(LABELS), 1, i)
    ax.hist(scores[lab][scores[lab] > 0], bins=20, range=(0, 1))
    ax.set_title(lab)
plt.tight_layout()
plt.show()
