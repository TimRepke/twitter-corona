import re
from datetime import datetime
from dataclasses import dataclass, field
import sqlite3
import os
from itertools import chain
from functools import reduce
from collections import defaultdict, Counter

import streamlit as st
from matplotlib import pyplot as plt
import pickle
import scipy.spatial.distance
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Union
import numpy as np
import plotly.graph_objs as go
from plotly.offline.offline import plot as off_plot
from utilities import Tweets, Tweet, GroupedHashtags

def pyplot_similarity_matrix(sim):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    plt.pcolormesh(sim, cmap=plt.cm.ocean)
    plt.colorbar()
    fig.patch.set_facecolor('#FFFFFF')
    plt.show()


def plotly_similarity_matrix(sim, labels, filename):
    trace = go.Heatmap(z=sim, x=labels, y=labels)
    off_plot([trace], filename=filename)


if __name__ == '__main__':
    tweets = Tweets(db_file='identifier.sqlite')

    print('Grouping tweets...')
    vectoriser = CountVectorizer(min_df=2, max_df=0.8)
    grouped_hashtags = GroupedHashtags(tweets.groupby_date('%Y-%W'), vectoriser=vectoriser)
    group_similarities = grouped_hashtags.pairwise_similarities(metric='jaccard')

    with st.expander('Vectoriser Settings'):
        st.selectbox('Vectoriser Type', )

    for w, hts in weekly_hashtags.items():
        print(f'{w}: {hts.most_common(3)}')


    pyplot_similarity_matrix(group_similarities)
    plotly_similarity_matrix(
        group_similarities,
        [g + ','.join(l) for g, l in grouped_hashtags.most_common(3, include_count=False, include_hashtag=True)],
        'data/jacc_similarities.html')
