import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from colorcet import glasbey
from collections import Counter
from datetime import datetime
from typing import Literal
import json

from utils.tweets import clean_tweet
from utils import load_embedded_data_jsonl
from utils.embedding import SentenceTransformerBackend
from utils.topics.frankentopic import FrankenTopic, UMAPArgs, VectorizerArgs, TSNEArgs, KMeansArgs, HDBSCANArgs
from utils.topics.utils import FrankenTopicUtils

# DATASET = 'geoengineering'
DATASET = 'climate'
LIMIT = 10000

# EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_MODEL = 'vinai/bertweet-large'
EMB_INCLUDE_HASHTAGS = True

SOURCE_FILE = f'data/{DATASET}/tweets_sentiment_{LIMIT}.jsonl'
EMBEDDING_FILE = f'data/{DATASET}/tweets_embeddings_{LIMIT}_{EMB_INCLUDE_HASHTAGS}_' \
                 f'{EMBEDDING_MODEL.replace("/", "_")}.npy'

if __name__ == '__main__':
    print('Loading tweets...')
    with open(SOURCE_FILE) as f_in:
        tweets = [json.loads(l) for l in f_in]

    print('Loading embeddings...')
    embeddings = np.load(EMBEDDING_FILE)

    LAYOUT_ARGS = [
        UMAPArgs(
            spread=1.,
            min_dist=0.05,
            n_neighbors=50,
            densmap=False,
            set_op_mix_ratio=0.5
        ),
        TSNEArgs(
            perplexity=50,
            metric='cosine',
            dof=0.6,
            initialization='pca'
        )][1]

    CLUSTERING_ARGS = [
        KMeansArgs(max_n_topics=40,
                   min_docs_per_topic=int((len(tweets) / 40) / 3)),
        HDBSCANArgs(
            # for 1k
            # min_samples=5,
            # min_cluster_size=20,
            # for 10k
            min_samples=15,
            min_cluster_size=75,
            # for 30k
            # min_samples=20,
            # min_cluster_size=100,
            # for 100k
            # min_samples=50,
            # min_cluster_size=500,
            cluster_selection_epsilon=0.
        )
    ][1]

    topic_model = FrankenTopic(
        cluster_args=CLUSTERING_ARGS,
        n_words_per_topic=15,
        n_candidates=200,
        mmr_diversity=0.1,
        vectorizer_args=VectorizerArgs(max_df=.7, stop_words='english'),
        dr_args=LAYOUT_ARGS,
        emb_backend=SentenceTransformerBackend,
        emb_model='paraphrase-multilingual-MiniLM-L12-v2',
        cache_layout=f'data/{DATASET}/layout_{LIMIT}.npy'
    )
    topic_model.fit([clean_tweet(t['text']) for t in tweets], embeddings)

    topic_model_utils = FrankenTopicUtils(tweets=tweets,
                                          topic_model=topic_model,
                                          n_tokens_per_topic=20)

    # Write topic words to console
    topic_model_utils.list_topics(emotions_model='bertweet-sentiment',
                                  emotions_keys=['negative', 'neutral', 'positive'])

    fig = topic_model_utils.landscape(
        emotions_model='betweet-sentiment',
        emotions_keys=['negative', 'positive'],
        emotions_colours=['Reds', 'Greens'],
        keyword_source='mmr',
        n_keywords_per_topic_legend=3,
        n_keywords_per_topic_map=6,
        colormap=glasbey
    )
    fig.write_html(f'data/{DATASET}/landscape_{LIMIT}.html')
    # fig.write_image('data/plt_emotions_static.png')

    DATE_FORMAT: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'
    fig = topic_model_utils.temporal_stacked(date_format=DATE_FORMAT,
                                             n_keywords_per_topic=5,
                                             keyword_source='mmr',
                                             colorscheme=glasbey)
    fig.write_html(f'data/{DATASET}/temporal_topics_stacked_{LIMIT}_{DATE_FORMAT}.html')

    # fig.write_image(f'data/temporal_topics_stacked_{DATASET}_{LIMIT}_{DATE_FORMAT}.png')

    # fig = go.Figure()
    # fig = topic_model_utils.temporal_hist(date_format=DATE_FORMAT,
    #                                              n_keywords_per_topic=5,
    #                                              keyword_source='mmr',
    #                                              colorscheme=glasbey)
    # fig.write_html(f'data/temporal_topics_hist_{DATASET}_{LIMIT}_{SELECTED_FORMAT}.html')
    # fig.write_image(f'data/temporal_topics_hist_{DATASET}_{LIMIT}_{SELECTED_FORMAT}.png')
