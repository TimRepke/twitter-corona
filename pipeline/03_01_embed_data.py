import json
import os
from typing import Optional, Union
import argparse
import numpy as np
from utils.models import ModelCache, SentenceTransformerBackend, AutoModelBackend
from utils.io import exit_if_exists
from utils.cluster import ClusterJob, FileHandler, Config as SlurmConfig

parser = argparse.ArgumentParser(description='Tweet Embedder')
parser.add_argument('--model', choices=['minilm', 'bertopic'], type=str, required=False, default='minilm',
                    help='The embedding model to use.')
parser.add_argument('--mode', choices=['local', 'cluster'], type=str, required=False, default='local',
                    help='Whether to submit this as a cluster job or run locally.')
parser.add_argument('--upload-data', action='store_true',
                    help='Set this flag to force data upload to cluster')
parser.add_argument('--upload-model', action='store_true',
                    help='Set this flag to force model upload to cluster')
parser.add_argument('--init-cluster', action='store_true',
                    help='Set this flag to initialise the cluster environment')
parser.add_argument('--cluster-mail', type=str, required=False)
parser.add_argument('--cluster-user', type=str, required=False)
parser.add_argument('--cluster-jobname', type=str, required=False, default='twitter-embed')
parser.add_argument('--cluster-workdir', type=str, required=False, default='twitter')
args = parser.parse_args()


def clean_clean_text(txt):
    return txt.replace('MENTION', '').replace('URL', '').replace('HASHTAG', '')


def line2txt_hashtags(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text']) + (' '.join(tweet['meta']['hashtags']))


def line2txt_clean(line):
    tweet = json.loads(line)
    return clean_clean_text(tweet['clean_text'])


def embed_tweets(
        dataset: str,
        model: Union[SentenceTransformerBackend, AutoModelBackend],
        limit: int,
        include_hashtags: bool,
        verbose: bool = True,
        source_f: Optional[str] = None,
        target_f: Optional[str] = None):
    if source_f is None:
        source_f = f'data/{dataset}/tweets_filtered_{limit}.jsonl'
    if target_f is None:
        target_f = f'data/{dataset}/tweets_embeddings_{limit}_{include_hashtags}_{model.model_name}.npy'

    exit_if_exists(target_f)

    print('Loading texts...')
    with open(source_f) as f_in:
        if include_hashtags:
            texts = [line2txt_hashtags(l) for l in f_in]
        else:
            texts = [line2txt_clean(l) for l in f_in]

    print('Embedding texts...')
    embeddings = model.embed_documents(texts, verbose=verbose)

    print('Storing embeddings...')
    np.save(target_f, embeddings)


if __name__ == '__main__':
    model_cache_location = 'data/models/'
    if args.mode == 'cluster':
        assert args.cluster_user is not None or 'You need to set --cluster-user'
        assert args.cluster_mail is not None or 'You need to set --cluster-mail'
        s_config = SlurmConfig(username=args.cluster_user,
                               email_address=args.cluster_mail,
                               jobname=args.cluster_jobname,
                               workdir=args.cluster_workdir,
                               memory='20G',
                               partition='gpu',
                               time_limit='4:00:00',
                               env_vars_run={
                                   'OPENBLAS_NUM_THREADS': 1,
                                   'TRANSFORMERS_OFFLINE': 1
                               })
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'],
                                   model_cache=model_cache_location,
                                   required_models=[args.model])
        s_job = ClusterJob(config=s_config,
                           main_script='pipeline/03_01_embed_data.py',
                           script_params={
                               'mode': 'local',
                               'model': args.model
                           })
        if args.init_cluster:
            s_job.initialise(file_handler)
        # else:
        #     file_handler.sync_code()
        s_job.submit_job()
    else:
        model_cache = ModelCache(model_cache_location)
        embedding_model = model_cache.get_embedder(args.model)
        embed_tweets(
            dataset='climate2',
            model=embedding_model,
            limit=10000,
            include_hashtags=True,
        )
