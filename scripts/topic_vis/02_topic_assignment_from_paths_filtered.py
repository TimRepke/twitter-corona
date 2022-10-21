from scripts.util import read_supertopics, SuperTopic, get_spottopics, DateFormat, read_temp_dist, smooth
from typing import Literal, Optional
import numpy as np
from matplotlib import pyplot as plt
import re
import seaborn as sns
import csv
from itertools import chain, repeat
import pandas as pd
from sklearn.neighbors import KernelDensity
from multiprocessing import Pool
import json
from matplotlib.patches import Polygon
import matplotlib.path as mplPath
from tqdm import tqdm

FILE_TSNE = 'data/geoengineering/layout_tsne.npy'
FILE_TSNE_FILTERED = 'data/geoengineering/layout_tsne_filtered.npy'
FILE_TWEETS = 'data/geoengineering/tweets_classified2.jsonl'

print('Reading tsne...')
with open(FILE_TSNE, 'rb') as f:
    TSNE_FULL = np.load(f)
EPS = 1e-12

filtered = {'c_01', 'c_02', 'c_03', 'c_04', 'c_05', 'c_06', 'c_07', 'c_08', 'c_55', 'c_09'}
with open(FILE_TWEETS) as f:
    mask = [json.loads(line)['sid'][0] in filtered for line in tqdm(f)]

TSNE_FILTERED = TSNE_FULL[mask]
with open(FILE_TSNE_FILTERED, 'wb') as f:
    np.save(f, TSNE_FILTERED)

paths_finn = [
    [[-17.380645161290325, 4.935064935064933], [-14.283870967741933, 5.714285714285712],
     [-13.27741935483871, 3.3116883116883145], [-16.761290322580646, 2.2077922077922096]],
    [[-16.606451612903225, 1.8831168831168839], [-15.290322580645162, -4.09090909090909],
     [-8.941935483870967, -6.1688311688311686], [-7.393548387096775, 1.9480519480519476]],
    [[-5.148387096774194, 1.558441558441558], [-5.845161290322579, -4.285714285714288],
     [1.0451612903225822, -4.870129870129869], [5.3806451612903246, 0.779220779220779],
     [0.5806451612903203, 2.597402597402599]],
    [[5.845161290322579, -2.337662337662337], [4.6064516129032285, -9.090909090909086],
     [9.79354838709677, -10.38961038961039], [13.277419354838713, -4.350649350649348],
     [10.335483870967742, -0.9090909090909101]],
    [[12.116129032258065, -8.506493506493506], [9.174193548387102, -12.272727272727273],
     [13.277419354838713, -13.766233766233768], [20.245161290322578, -11.363636363636367]],
    [[12.270967741935479, -2.27272727272727], [16.529032258064518, -0.2597402597402585],
     [20.245161290322578, -3.6363636363636367], [14.283870967741933, -6.493506493506494]],
    [[2.516129032258064, 3.1818181818181834], [8.477419354838709, 0.19480519480519476],
     [13.277419354838713, 4.0909090909090935], [11.729032258064514, 5.2597402597402585],
     [5.225806451612897, 5.064935064935064]],
    [[5.303225806451614, 5.324675324675322], [4.451612903225808, 9.675324675324676],
     [10.180645161290322, 13.181818181818182], [14.05161290322581, 10.25974025974026],
     [11.806451612903224, 5.454545454545453]],
    [[12.580645161290327, 5.324675324675322], [19.470967741935482, 16.493506493506494],
     [26.593548387096774, 8.376623376623378], [21.716129032258067, -3.051948051948049],
     [13.509677419354844, 3.246753246753247], [13.819354838709678, 4.415584415584416]],
    [[4.219354838709677, 10.454545454545455], [-4.06451612903226, 15.064935064935064],
     [6.232258064516131, 23.246753246753247], [17.76774193548387, 19.415584415584416],
     [14.670967741935485, 10.974025974025976], [10.258064516129032, 13.506493506493507]],
    [[3.6774193548387117, 10.129870129870131], [4.761290322580642, 5.649350649350648],
     [1.8967741935483886, 3.376623376623378], [-5.070967741935483, 2.012987012987015],
     [-11.961290322580645, 3.0519480519480524], [-14.129032258064516, 12.597402597402597],
     [-5.690322580645162, 15.324675324675324]],
    [[9.406451612903226, 0.2597402597402585], [12.735483870967741, 2.9220779220779214], [15.522580645161291, 0],
     [12.348387096774196, -1.8181818181818201]]
]
paths = paths_finn

FILE_OUT = 'data/geoengineering/topics_finn3.npy'
polys = [mplPath.Path(np.array(p)) for p in paths]


def get_poly(pt):
    for pi, poly in enumerate(polys, start=1):
        if poly.contains_point(pt):
            return pi
    return 0


assignments = [get_poly(p) for p in tqdm(TSNE_FILTERED)]
with open(FILE_OUT, 'wb') as f_out:
    np.save(f_out, np.array(assignments))