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
from matplotlib import rcParams, cycler
from matplotlib import cm

FILE_TSNE = 'data/geoengineering/layout_tsne.npy'
FILE_TWEETS = 'data/geoengineering/tweets_classified2.jsonl'
FILE_OUT = 'data/geoengineering/landscape.json'
colours = [list(col) for col in cm.tab20.colors]
print(colours[0])
print('Reading tsne...')
with open(FILE_TSNE, 'rb') as f:
    TSNE_FULL = np.load(f)

EPS = 1e-12

print('Find plot dimensions')
xmin = TSNE_FULL[:, 0].min()
xmax = TSNE_FULL[:, 0].max()
ymin = TSNE_FULL[:, 1].min()
ymax = TSNE_FULL[:, 1].max()
xbins = 100j
ybins = 100j
xx, yy = np.mgrid[xmin:xmax:xbins, ymin:ymax:ybins]
xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T

qmap = {'Geoengineering': ['g_01', 'g_02', 'g_05', 'g_06'],
        'SRM': ['s_01', 's_02', 's_21', 's_22', 's_27'],
        'Aerosols': ['s_04', 's_07'],#Aerosol Injection
        'Brightening': ['s_09', 's_20', 's_23', 's_24'],#Cloud Brightening
        'Albedo': ['s_10', 's_11', 's_25', 's_30'],  # Surface Albedo Modification
        'Thinning': ['s_12', 's_14'], #Cloud Thinning
        'Space': ['s_16', 's_17', 's_18', 's_19', 's_29'], #Space Shades
        'GGR': ['c_01', 'c_02', 'c_03', 'c_04', 'c_05', 'c_06', 'c_07', 'c_08', 'c_55', 'c_09'],
        'CCS': ['c_10', 'c_11', 'c_12', 'c_13', 'c_14', 'c_15', 'c_16', 'c_17'],
        'Methane': ['c_18', 'c_19', 'c_54'], #Methane Removal
        'Oceans': ['c_20', 'c_21', 'c_22', 'c_50'],#Ocean Fertilization
        'EW': ['c_23', 'c_49', 'c_24', 'c_25', 'c_26', 'c_51'],  # Enhanced Weathering
        'Biochar': ['c_27'],
        'Forests': ['c_29', 'c_30', 'c_31', 'c_32'],  # Afforestation and Reforestation
        'SCS': ['c_33', 'c_36', 'c_37'],  # Soil Carbon Sequestration
        'BECCS': ['c_38', 'c_39', 'c_40'],
        'Blue Carbon': ['c_41', 'c_42', 'c_43', 'c_52', 'c_53', 'c_44'],
        'DAC': ['c_45', 'c_46', 'c_47', 'c_48']  # Direct Air Capture
        }
qmap_i = {q: g for g, qs in qmap.items() for q in qs}

with open(FILE_TWEETS) as f:
    queries = [qmap_i[json.loads(line)['sid'][0]] for line in tqdm(f)]

unique_queries = set(queries)
print(unique_queries)
uq_map = {k: i for i, k in enumerate(unique_queries)}
queries = np.array([uq_map[q] for q in queries])

print('Computing densities...')
densities = {}
for group, gi in uq_map.items():
    print(f'  > {group}')
    st_data = TSNE_FULL[queries == gi]

    x = st_data[:, 0]
    y = st_data[:, 1]
    xy_train = np.vstack([y, x]).T

    kde = KernelDensity(kernel='exponential', metric='euclidean',
                        bandwidth=1.1, atol=0.0005, rtol=0.01)
    kde.fit(xy_train)
    n_threads = 30
    with Pool(n_threads) as p:
        z = np.concatenate(p.map(kde.score_samples, np.array_split(xy_sample, n_threads)))

    z = np.exp(z)
    zz = np.reshape(z, xx.shape)
    densities[group] = zz

print('Plotting...')
plt.figure(figsize=(15, 15), dpi=150)
for group, gi in uq_map.items():
    st_data = TSNE_FULL[queries == gi]

    x = st_data[:, 0]
    y = st_data[:, 1]
    plt.scatter(x, y, marker='X', alpha=0.1, s=0.1, c=[colours[gi]])
    c = plt.contour(xx, yy, densities[group], 3, colors=[colours[gi]])
    fmt = {}
    for l in c.levels:
        fmt[l] = group
    plt.clabel(c, c.levels, fmt=fmt, inline=True, fontsize=10, colors='black')

# plt.figure(figsize=(15, 15), dpi=150)
# for q, i in uq_map.items():
#     plt.scatter(TSNE_FULL[queries == i][:, 0], TSNE_FULL[queries == i][:, 1], marker='X', alpha=0.1, s=0.1, label=q)

# plt.ylim(-25, 25)
# plt.xlim(-30, 30)
plt.ylim(-18, 18)
plt.xlim(-20, 25)
plt.tight_layout()
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
# plt.legend()
plt.show()
