import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from colorcet import glasbey
from collections import Counter
from datetime import datetime
from typing import Literal

from utils.tweets import clean_tweet
from utils import load_embedded_data_jsonl
from utils.embedding import SentenceTransformerBackend
from utils.topics.frankentopic import FrankenTopic, UMAPArgs, VectorizerArgs, TSNEArgs, KMeansArgs, HDBSCANArgs
from utils.topics.utils import FrankenTopicUtils

SOURCE_FILES = {
    'geo': 'data/geoengineering_tweets_sentop4_full.jsonl',
    'climate': 'data/climate_tweets_sentiment.jsonl'
}
DATASET = 'climate'
SOURCE_FILE = SOURCE_FILES[DATASET]

LIMIT = 10000

if __name__ == '__main__':
    embedding_backend = SentenceTransformerBackend
    # embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
    embedding_model = 'vinai/bertweet-large'

    tweets, embeddings = load_embedded_data_jsonl(
        source_file=SOURCE_FILE,
        cache_dir='data/',
        cache_prefix=DATASET,
        limit=LIMIT,
        remove_urls=True,
        remove_nonals=True,
        remove_hashtags=True,
        remove_mentions=True,
        backend=embedding_backend,
        model=embedding_model)

    umap_params = UMAPArgs(
        spread=1.,
        min_dist=0.05,
        n_neighbors=50,
        densmap=False,
        set_op_mix_ratio=0.5
    )
    tsne_params = TSNEArgs(
        perplexity=50,
        metric='cosine',
        dof=0.6,
        initialization='pca'
    )

    kmeans_args = KMeansArgs(max_n_topics=40,
                             min_docs_per_topic=int((len(tweets) / 40) / 3))
    hdbscan_args = HDBSCANArgs(
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
    topic_model = FrankenTopic(
        # cluster_args=hdbscan_args,
        cluster_args=kmeans_args,
        n_words_per_topic=15,
        n_candidates=200,
        mmr_diversity=0.1,
        vectorizer_args=VectorizerArgs(max_df=.7, stop_words='english'),
        dr_args=tsne_params,
        emb_backend=embedding_backend,
        # emb_model=embedding_model,
        emb_model='paraphrase-multilingual-MiniLM-L12-v2',
        cache_layout=f'data/layout_{DATASET}_emo_{LIMIT}.npy'
    )
    topic_model.fit([clean_tweet(t['text']) for t in tweets], embeddings)

    topic_model_utils = FrankenTopicUtils(tweets=tweets,
                                          topic_model=topic_model,
                                          n_tokens_per_topic=20)

    # Write topic words to console
    topic_model_utils.list_topics(emotions_model='bertweet-sentiment',
                                  emotions_keys=['negative', 'neutral', 'positive'])

    emo_model = 'bertweet-sentiment'
    emo_labels = ['negative', 'neutral', 'positive']
    emo_labels_heat = [('negative', 'Reds'), ('positive', 'Greens')]  # + neutral

    print(f'Using {emo_model} for sentiment/emotion annotations.')
    fig = topic_model_utils.landscape(
        emotions_model='betweet-sentiment',
        emotions_keys=['negative', 'positive'],
        emotions_colours=['Reds', 'Greens'],
        keyword_source='mmr',
        n_keywords_per_topic_legend=3,
        n_keywords_per_topic_map=6,
        colormap=glasbey
    )
    fig.write_html(f'data/landscape_{DATASET}_{LIMIT}.html')
    # fig.write_image('data/plt_emotions_static.png')

    DATE_FORMAT: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'
    fig = topic_model_utils.temporal_stacked(date_format=DATE_FORMAT,
                                             n_keywords_per_topic=5,
                                             keyword_source='mmr',
                                             colorscheme=glasbey)
    fig.write_html(f'data/temporal_topics_stacked_{DATASET}_{LIMIT}_{DATE_FORMAT}.html')
    fig.write_image(f'data/temporal_topics_stacked_{DATASET}_{LIMIT}_{DATE_FORMAT}.png')

    # fig = go.Figure()
    # fig = topic_model_utils.temporal_hist(date_format=DATE_FORMAT,
    #                                              n_keywords_per_topic=5,
    #                                              keyword_source='mmr',
    #                                              colorscheme=glasbey)
    # fig.write_html(f'data/temporal_topics_hist_{DATASET}_{LIMIT}_{SELECTED_FORMAT}.html')
    # fig.write_image(f'data/temporal_topics_hist_{DATASET}_{LIMIT}_{SELECTED_FORMAT}.png')
