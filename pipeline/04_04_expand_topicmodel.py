from typing import Optional, Literal, List
import os
from utils.cluster import ClusterJobBaseArguments
from copy import deepcopy


class ExpandTopicModelArgs(ClusterJobBaseArguments):
    model: Literal['minilm', 'bertopic'] = 'minilm'  # The embedding model to use.
    model_cache: str = 'data/models/'  # Location for cached models.

    file_embeddings: Optional[str] = None  # The file containing embedded tweets (relative to source root)
    file_dump: Optional[str] = None  # The file containing topic model dump (relative to source root)
    file_tweets: Optional[str] = None  # The file containing unfiltered cleaned tweets (relative to source root)
    output_directory: Optional[str] = None  # The directory to write outputs to

    limit: Optional[int] = 10000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    projection: Literal['umap', 'tsne'] = 'tsne'  # The dimensionality reduction method to use
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    n_tokens_listing: int = 20  # number of tokens per topic to show in listing and store in dump
    n_tokens_candidates: int = 150  # number of tokens to internally represent topics
    n_tokens_landscape: int = None  # number of tokens per topic in landscape rendering
    n_tokens_plot: int = 5
    mmr_diversity: float = 0.3  # diversity for MMR resampling
    temporal_grouping: Literal['monthly', 'yearly', 'weekly', 'daily'] = 'monthly'  # grouping for temporal topic dist

    filter_only_en: bool = True
    filter_allow_lang_null: bool = False
    filter_min_tokens: int = 4
    filter_max_hashtags: int = 5
    filter_from_date: str = '2018-01'
    filter_to_date: str = '2020-12'

    cluster_jobname: str = 'twitter-expand-topics'
    cluster_workdir: str = 'twitter'


if __name__ == '__main__':
    args = ExpandTopicModelArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = ExpandTopicModelArgs().load(args.args_file)

    _include_hashtags = not args.excl_hashtags
    target_dir = args.output_directory or f'data/{args.dataset}/topics/'
    file_embeddings = args.file_embeddings or \
                      f'data/{args.dataset}/tweets_embeddings_{args.limit}_{_include_hashtags}_{args.model}.npy'
    file_dump = os.path.join(target_dir, f'dump_{args.limit}_{args.temporal_grouping}.json')
    file_tweets = args.file_tweets or f'data/{args.dataset}/tweets_clean.jsonl'

    if args.mode == 'cluster':

        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler

        s_config = SlurmConfig.from_args(args)
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'],
                                   model_cache=args.model_cache,
                                   required_models=[args.model])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)

        cluster_args = deepcopy(args)
        cluster_args.mode = 'local'
        cluster_args.file_layout = os.path.join(s_config.datadir_path, file_layout)
        cluster_args.file_tweets = os.path.join(s_config.datadir_path, file_tweets)
        cluster_args.output_directory = os.path.join(s_config.datadir_path, target_dir)
        s_job.submit_job(main_script='pipeline/04_04_expand_topicmodel.py', params=cluster_args)

    else:
        import json
        import numpy as np
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        from sklearn.feature_extraction.text import TfidfVectorizer
        from utils.topics.frankentopic import get_top_mmr, get_top_tfidf
        from utils.topics.utils import simple_topic_listing, get_temporal_distribution, get_tokens, date2group
        from utils.tweets import FilterResult, TweetFilter
        import plotly.graph_objects as go
        from colorcet import glasbey

        print('Loading layout...')
        layout = np.load(file_layout)

        print('Clustering with HDBSCAN...')
        labels = get_cluster_labels(layout, min_samples=args.min_samples, min_cluster_size=args.min_cluster_size,
                                    cluster_selection_epsilon=args.cluster_selection_epsilon, alpha=args.alpha,
                                    cluster_selection_method=args.cluster_selection_method)
        topic_ids = np.unique(labels)

        print('Loading tweets...')
        with open(file_tweets) as f_in:
            tweets = [json.loads(line) for line in f_in]

        print('Grouping tweets...')
        grouped_texts = [
            [tweets[i]['clean_text'] for i in np.argwhere(labels == label).reshape(-1, )]
            for label in np.unique(labels)
        ]

        print('Vectorising groups...')
        stop_words = list(ENGLISH_STOP_WORDS) + ['url', 'mention', 'hashtag', 'rt']
        vectorizer = TfidfVectorizer(max_df=0.7, stop_words=stop_words, ngram_range=(1, 2),
                                     min_df=0, lowercase=True, use_idf=True, smooth_idf=True)
        tf_idf_vecs = vectorizer.fit_transform([' '.join(g) for g in grouped_texts])
        vocab = {v: k for k, v in vectorizer.vocabulary_.items()}

        # compute topic representations
        topics_tfidf = get_top_tfidf(vectors=tf_idf_vecs, token_lookup=vocab, n_tokens=args.n_tokens_candidates)
        topics_mmr = get_top_mmr(topics_tfidf, n_tokens=args.n_tokens_listing, mmr_diversity=args.mmr_diversity,
                                 model_cache_location=args.model_cache, model=args.model)

        # print topics to console
        simple_topic_listing(topics_tfidf=topics_tfidf, topics_mmr=topics_mmr,
                             n_tokens=args.n_tokens_listing, labels=labels)

        distributions, time_groups_ = write_distributions(boosts=[[],
                                                                  ['retweets'],
                                                                  ['replies'],
                                                                  ['likes'],
                                                                  ['retweets', 'likes'],
                                                                  ['retweets', 'likes', 'replies']],
                                                          norms=['abs', 'row', 'col'])

        dump = get_dump()
        with open(os.path.join(args.output_directory, f'dump_{args.limit}_{args.temporal_grouping}.json'), 'w') as f:
            f.write(json.dumps(dump))
