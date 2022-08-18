from scripts.util import DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import re
import seaborn as sns
import csv
from itertools import chain, repeat
import pandas as pd
from sklearn.neighbors import KernelDensity
from multiprocessing import Pool
from functools import reduce
import imageio
import json
from collections import defaultdict

FILE_TWEETS = 'data/geoengineering/tweets_classified2.jsonl'
FILE_LABELS = 'data/geoengineering/topics_finn2.npy'
FILE_TSNE = 'data/geoengineering/layout_tsne.npy'

print('Reading tsne...')
with open(FILE_TSNE, 'rb') as f:
    TSNE = np.load(f)

print('Reading labels...')
with open(FILE_LABELS, 'rb') as f:
    LABELS = np.load(f)

print('Reading timestamps...')
with open(FILE_TWEETS, 'r') as f:
    TIMESTAMPS = [json.loads(line)['created_at'][:19] for line in f]

print('Sorting timestamps...')
SORTED = list(sorted(enumerate(TIMESTAMPS), key=lambda d: d[1]))

print('Aggregating quarters...')
QUARTERS = defaultdict(list)
for idx, timestamp in SORTED:
    quarter = f'{timestamp[:4]}'  # -Q{((int(timestamp[5:7]) - 1) // 3) + 1}'
    QUARTERS[quarter].append(idx)

EPS = 1e-12

print('Find plot dimensions')
xmin = -20
xmax = 25
ymin = -18
ymax = 18
xbins = 100j
ybins = 100j

FRAMES = sorted(QUARTERS.keys())
print(FRAMES)
print(len(FRAMES))

INCLUDE_PRE = False


def draw_frame(frame):
    quarter_i, quarter_key = frame

    fig = plt.figure(figsize=(10, 10), dpi=100)
    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)
    frame_indices = QUARTERS[quarter_key]

    frame_tsne = TSNE[frame_indices]

    plt.title(f'{quarter_key} ({len(frame_indices):,}tweets)')
    if quarter_i > 0 and INCLUDE_PRE:
        pre_frame_indices = [idx_ for i in range(quarter_i) for idx_ in QUARTERS[FRAMES[i]]]
        pre_frame_tsne = TSNE[pre_frame_indices]
        plt.scatter(pre_frame_tsne[:, 0], pre_frame_tsne[:, 1], marker='X', alpha=0.05, c='gray', s=0.05)
    plt.scatter(frame_tsne[:, 0], frame_tsne[:, 1], marker='X', alpha=0.1, c='red', s=0.1)

    plt.tight_layout()
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


print('Plotting...')
drawn_frames = [draw_frame(i) for i in tqdm(enumerate(FRAMES))]
print('Making GIF...')
imageio.mimsave('data/geoengineering/topics_finn2/landscape_yearly.gif', drawn_frames, fps=2)
imageio.mimsave('data/geoengineering/topics_finn2/landscape_yearly.mp4', drawn_frames, fps=2)
