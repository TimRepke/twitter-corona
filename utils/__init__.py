import os
import numpy as np
from typing import Type

from .tweets import Tweets
from .embedding import SentenceTransformerBackend, BaseEmbedder


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
