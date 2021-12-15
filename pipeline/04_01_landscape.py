import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from colorcet import glasbey
from collections import Counter
from datetime import datetime
from typing import Literal
import json
import os

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

TARGET_DIR = f'data/{DATASET}/topics'
os.makedirs(TARGET_DIR, exist_ok=True)

DATE_FORMAT: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'

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
            perplexity=40,
            early_exaggeration=20,
            metric='cosine',
            dof=0.6,
            initialization='pca'
        )][1]

    CLUSTERING_ARGS = [
        KMeansArgs(max_n_topics=40,
                   min_docs_per_topic=int((len(tweets) / 40) / 3)),
        HDBSCANArgs(
            min_samples=8,
            min_cluster_size=30,
            cluster_selection_epsilon=0.,
            alpha=1.,
            cluster_selection_method='eom'
        )
    ][1]

    topic_model = FrankenTopic(
        cluster_args=CLUSTERING_ARGS,
        n_words_per_topic=15,
        n_candidates=200,
        mmr_diversity=0.1,
        vectorizer_args=VectorizerArgs(max_df=.7, stop_words='english', ngram_range=(1, 2)),
        dr_args=LAYOUT_ARGS,
        emb_backend=SentenceTransformerBackend,
        emb_model='paraphrase-multilingual-MiniLM-L12-v2',
        cache_layout=f'{TARGET_DIR}/layout_{LIMIT}_{LAYOUT_ARGS.__class__.__name__}.npy'
    )
    topic_model.fit([clean_tweet(t['text']) for t in tweets], embeddings)

    topic_model_utils = FrankenTopicUtils(tweets=tweets,
                                          topic_model=topic_model,
                                          n_tokens_per_topic=20)

    # Write topic words to console
    topic_model_utils.list_topics(emotions_model='bertweet-sentiment',
                                  emotions_keys=['negative', 'neutral', 'positive'],
                                  include_mmr=False)

    fig = topic_model_utils.landscape(
        # emotions_model='bertweet-sentiment',
        # emotions_keys=['negative', 'positive'],
        # emotions_colours=['Reds', 'Greens'],
        keyword_source='tfidf',
        n_keywords_per_topic_legend=6,
        n_keywords_per_topic_map=4,
        colormap=glasbey
    )
    fig.write_html(f'{TARGET_DIR}/landscape_{LIMIT}.html')
    # fig.write_image('data/plt_emotions_static.png')

    fig = topic_model_utils.temporal_stacked_fig(date_format=DATE_FORMAT,
                                                 n_keywords_per_topic=5,
                                                 keyword_source='tfidf',
                                                 colorscheme=glasbey)
    fig.write_html(f'{TARGET_DIR}/temporal_topics_stacked_{LIMIT}_{DATE_FORMAT}.html')

    for boost in [[], ['retweets'], ['replies'], ['likes'], ['retweets', 'likes']]:
        for norm in ['abs', 'row', 'col']:
            temporal_topics = topic_model_utils.get_temporal_distribution(date_format=DATE_FORMAT,
                                                                          boost=boost, skip_topic_zero=True)
            time_groups = sorted(temporal_topics.keys())

            topic_dist = np.array([temporal_topics[tg].tolist() for tg in time_groups])

            fig = go.Figure([go.Bar(x=[f'd:{d}' for d in time_groups],
                                    y=topic_dist.sum(axis=0))])
            fig.write_html(f'{TARGET_DIR}/histogram_{LIMIT}_{DATE_FORMAT}_{"_".join(boost)}.html')

            if norm == 'row':
                topic_dist = topic_dist / (topic_dist.sum(axis=0) + 0.0000001)
            elif norm == 'col':
                topic_dist = (topic_dist.T / (topic_dist.sum(axis=1) + 0.0000001)).T
            # elif norm == 'abs':
            #     pass

            fig = go.Figure(data=go.Heatmap(
                z=topic_dist.T,
                x=[f'd:{d}' for d in time_groups],
                y=[' '.join(topic_model_utils.get_keywords(t, keyword_source='mmr', n_keywords=4))
                   for t in topic_model_utils.topic_ids[1:]],
                hoverongaps=False))
            fig.write_html(f'{TARGET_DIR}/temporal_{LIMIT}_{DATE_FORMAT}_{norm}_{"_".join(boost)}.html')

    # fig.write_image(f'data/temporal_topics_stacked_{DATASET}_{LIMIT}_{DATE_FORMAT}.png')

    # fig = go.Figure()
    # fig = topic_model_utils.temporal_hist(date_format=DATE_FORMAT,
    #                                              n_keywords_per_topic=5,
    #                                              keyword_source='mmr',
    #                                              colorscheme=glasbey)
    # fig.write_html(f'data/temporal_topics_hist_{DATASET}_{LIMIT}_{SELECTED_FORMAT}.html')
    # fig.write_image(f'data/temporal_topics_hist_{DATASET}_{LIMIT}_{SELECTED_FORMAT}.png')
