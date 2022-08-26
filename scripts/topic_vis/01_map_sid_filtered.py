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

print('Reading tsne...')
with open(FILE_TSNE, 'rb') as f:
    TSNE_FULL = np.load(f)

qmap = {'Geoengineering (general)': ['g_01', 'g_02', 'g_05', 'g_06'],
        'SRM (general)': ['s_01', 's_02', 's_21', 's_22', 's_27'],
        'Aerosol Injection': ['s_04', 's_07'],
        'Cloud Brightening': ['s_09', 's_20', 's_23', 's_24'],
        'Surface Albedo Modification': ['s_10', 's_11', 's_25', 's_30'],
        'Cloud Thinning': ['s_12', 's_14'],
        'Space Shades': ['s_16', 's_17', 's_18', 's_19', 's_29'],
        'GGR (general)': ['c_01', 'c_02', 'c_03', 'c_04', 'c_05', 'c_06', 'c_07', 'c_08', 'c_55', 'c_09'],
        'CCS': ['c_10', 'c_11', 'c_12', 'c_13', 'c_14', 'c_15', 'c_16', 'c_17'],
        'Methane Removal': ['c_18', 'c_19', 'c_54'],
        'Ocean Fertilization': ['c_20', 'c_21', 'c_22', 'c_50'],
        'Enhanced Weathering': ['c_23', 'c_49', 'c_24', 'c_25', 'c_26', 'c_51'],
        'Biochar': ['c_27'],
        'Afforestation and Reforestation': ['c_29', 'c_30', 'c_31', 'c_32'],
        'Soil Carbon Sequestration': ['c_33', 'c_36', 'c_37'],
        'BECCS': ['c_38', 'c_39', 'c_40'],
        'Blue Carbon': ['c_41', 'c_42', 'c_43', 'c_52', 'c_53', 'c_44'],
        'Direct Air Capture': ['c_45', 'c_46', 'c_47', 'c_48']}
filtered = {'c_01', 'c_02', 'c_03', 'c_04', 'c_05', 'c_06', 'c_07', 'c_08', 'c_55', 'c_09'}
qmap_i = {q: g for g, qs in qmap.items() for q in qs}

with open(FILE_TWEETS) as f:
    queries = [(qmap_i[tweet['sid'][0]], tweet['sid'][0]) for line in tqdm(f)
               if (tweet := json.loads(line)) is not None]

mask = np.array([q[1] in filtered for q in queries])
queries = np.array([q[0] for q in queries])

unique_queries = set(queries)
print(unique_queries)
uq_map = {k: i for i, k in enumerate(unique_queries)}
queries_u = np.array([uq_map[q] for q in queries])

plt.figure(figsize=(15, 15), dpi=150)
for q, i in uq_map.items():
    smask = mask & (queries_u == i)
    if sum(smask) > 0:
        plt.scatter(TSNE_FULL[smask][:, 0], TSNE_FULL[smask][:, 1], marker='X', alpha=0.1, s=0.1, label=q)
    else:
        print(f'not enough points for {q} ({i})')

plt.ylim(-25, 25)
plt.xlim(-30, 30)
plt.tight_layout()
plt.tight_layout()
# plt.legend()
plt.show()
