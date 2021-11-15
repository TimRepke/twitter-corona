import json
import os
import numpy as np
from typing import Type

from .tweets import Tweets, clean_tweet
from .embedding import SentenceTransformerBackend, BaseEmbedder


def load_embedded_data_jsonl(
        backend: Type[BaseEmbedder] = SentenceTransformerBackend,
        model: str = 'vinai/bertweet-large',
        cache_dir: str = 'data/',
        source_file: str = 'data/geoengineering_tweets_tweets.jsonl',
        limit: int = 1000,
        remove_urls: bool = True,
        remove_nonals: bool = True,
        remove_hashtags: bool = False,
        remove_mentions: bool = True) -> tuple[list[dict], np.ndarray]:
    cache_file = os.path.join(cache_dir,
                              f'emb_cache_{limit}_{remove_urls}_{remove_nonals}_{remove_hashtags}_{remove_mentions}_'
                              f'{backend.__name__}_{model.replace("/", "_")}.npy')

    print('Loading tweets...')
    with open(source_file) as f:
        num_lines = sum(1 for l in f)
        print(f'  - Source file contains {num_lines} tweets.')
    with open(source_file) as f:
        if num_lines <= limit:
            print('  - Loading everything...')
            tweets_ = [json.loads(l) for l in f]
        else:
            skip_lines = max(int(num_lines / limit), 1)
            print(f'  - Targeting to load {limit} tweets by reading every {skip_lines}th tweet...')
            tweets_ = [json.loads(l) for i, l in enumerate(f) if (i % skip_lines) == 0]
            limit = num_lines

    tweets_ = [t for t in tweets_ if t['text'] is not None and len(t['text']) > 5]
    print(f'  - Actually loaded {len(tweets_)} (post filtering empty tweets).')
    print(f'Assuming tweets are sorted by date, they range from '
          f'{tweets_[0]["created_at"]} to {tweets_[-1]["created_at"]}')

    print(f'Embedding cache file should be: {cache_file}')

    if os.path.isfile(cache_file):
        print('Loading embeddings from cache...')
        embeddings_ = np.load(cache_file)
    else:
        print('Embedding tweets...')
        model_ = backend(model)
        embeddings_ = model_.embed_documents([clean_tweet(t['text'],
                                                          remove_hashtags=remove_hashtags,
                                                          remove_nonals=remove_nonals,
                                                          remove_urls=remove_urls,
                                                          remove_mentions=remove_mentions) for t in tweets_],
                                             verbose=True)
        print('Storing embeddings...')
        np.save(cache_file, embeddings_)

    return tweets_, embeddings_


def load_embedded_data(backend: Type[BaseEmbedder] = SentenceTransformerBackend,
                       model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                       cache_dir: str = 'data/',
                       db_file: str = 'data/identifier.sqlite',
                       limit: int = 1000,
                       remove_urls: bool = True,
                       remove_nonals: bool = True,
                       remove_hashtags: bool = False,
                       remove_mentions: bool = True) -> tuple[Tweets, np.ndarray]:
    cache_file = os.path.join(cache_dir,
                              f'emb_cache_{limit}_{remove_urls}_{remove_nonals}_{remove_hashtags}_{remove_mentions}_'
                              f'{backend.__name__}_{model}.npy')

    print('Loading tweets...')
    tweets_ = Tweets(db_file=db_file,
                     limit=limit,
                     remove_urls=remove_urls,
                     remove_nonals=remove_nonals,
                     remove_hashtags=remove_hashtags,
                     remove_mentions=remove_mentions)

    print(f'Embedding cache file should be: {cache_file}')

    if os.path.isfile(cache_file):
        print('Loading embeddings from cache...')
        embeddings_ = np.load(cache_file)
    else:
        print('Embedding tweets...')
        model_ = backend(model)
        embeddings_ = tweets_.get_embeddings(model_)
        print('Storing embeddings...')
        np.save(cache_file, embeddings_)

    return tweets_, embeddings_
