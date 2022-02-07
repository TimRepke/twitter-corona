import json
import os
from typing import Optional, Union, Literal
import numpy as np
from utils.models import ModelCache, SentenceTransformerBackend, AutoModelBackend
from utils.io import exit_if_exists
from tap import Tap

PathLike = Union[str, bytes, os.PathLike]


class TweetEmbeddingArgs(Tap):
    model: Literal['minilm', 'bertopic'] = 'minilm'  # The embedding model to use.
    model_cache: str = 'data/models/'  # Location for cached models.
    file_in: Optional[str] = None  # The file to read data from (relative to source root)
    file_out: Optional[str] = None  # The file to write embeddings to (relative to source root)

    limit: Optional[int] = 10000  # Size of the dataset
    dataset: Optional[str] = 'climate2'  # Name of the dataset
    excl_hashtags: bool = False  # Set this flag to exclude hashtags in the embedding

    mode: Literal['local', 'cluster'] = 'local'  # Whether to submit this as a cluster job or run locally.

    cluster_mail: Optional[str] = None  # email address for job notifications
    cluster_user: Optional[str] = None  # PIK username
    cluster_time: Optional[str] = '4:00:00'  # Time limit for the cluster job
    cluster_ram: Optional[str] = '20G'  # Memory limit for the cluster job
    cluster_jobname: str = 'twitter-embed'
    cluster_workdir: str = 'twitter'

    upload_data: bool = False  # Set this flag to force data upload to cluster
    upload_model: bool = False  # Set this flag to force model upload to cluster
    cluster_init: bool = False  # Set this flag to initialise the cluster environment


def clean_clean_text(txt):
    return txt.replace('MENTION', '').replace('URL', '').replace('HASHTAG', '')


def line2txt_hashtags(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text']) + (' '.join(tweet['meta']['hashtags']))


def line2txt_clean(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text'])


def embed_tweets(
        source_f: PathLike,
        target_f: PathLike,
        model: Union[AutoModelBackend, SentenceTransformerBackend],
        include_hashtags: bool = True,
        verbose: bool = True):
    exit_if_exists(target_f)

    print('Loading texts...')
    with open(source_f) as f_in:
        if include_hashtags:
            texts = [line2txt_hashtags(line) for line in f_in]
        else:
            texts = [line2txt_clean(line) for line in f_in]

    print('Embedding texts...')
    embeddings = model.embed_documents(texts, verbose=verbose)

    print('Storing embeddings...')
    np.save(target_f, embeddings)


if __name__ == '__main__':
    args = TweetEmbeddingArgs(underscores_to_dashes=True).parse_args()

    _include_hashtags = not args.excl_hashtags

    if args.file_in is None:
        file_in = f'data/{args.dataset}/tweets_filtered_{args.limit}.jsonl'
    else:
        file_in = args.file_in
    if args.file_out is None:
        file_out = f'data/{args.dataset}/tweets_embeddings_{args.limit}_{_include_hashtags}_{args.model}.npy'
    else:
        file_out = args.file_out

    if args.mode == 'cluster':
        from utils.cluster import ClusterJob, FileHandler, Config as SlurmConfig

        assert args.cluster_user is not None or 'You need to set --cluster-user'
        assert args.cluster_mail is not None or 'You need to set --cluster-mail'

        s_config = SlurmConfig(username=args.cluster_user,
                               email_address=args.cluster_mail,
                               jobname=args.cluster_jobname,
                               workdir=args.cluster_workdir,
                               memory=args.cluster_ram,
                               partition='gpu',
                               time_limit=args.cluster_time,
                               env_vars_run={
                                   'OPENBLAS_NUM_THREADS': 1,
                                   'TRANSFORMERS_OFFLINE': 1
                               })
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'],
                                   model_cache=args.model_cache,
                                   required_models=[args.model],
                                   data_files=[file_in])
        s_job = ClusterJob(config=s_config,
                           main_script='pipeline/03_01_embed_data.py',
                           script_params={
                               'mode': 'local',
                               'model': args.model,
                               'model-cache': s_config.modeldir_path,
                               'file-in': os.path.join(s_config.datadir_path, file_in),
                               'file-out': os.path.join(s_config.datadir_path, file_out)
                           })
        if args.cluster_init:
            s_job.initialise(file_handler)
        else:
            if args.upload_model:
                file_handler.cache_upload_models()
            if args.upload_data:
                file_handler.upload_data()
            file_handler.sync_code()

        s_job.submit_job()
    else:
        _model_cache = ModelCache(args.model_cache)
        _model = _model_cache.get_embedder(args.model)

        embed_tweets(
            model=_model,
            source_f=file_in,
            target_f=file_out,
            include_hashtags=_include_hashtags,
        )
