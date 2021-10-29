import math
import os.path
import time

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from utils import load_embedded_data
from utils.embedding import SentenceTransformerBackend
from utils.topics import FrankenTopic, UMAPArgs, VectorizerArgs
from umap import UMAP

if __name__ == '__main__':
    embedding_backend = SentenceTransformerBackend
    embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
    tweets, embeddings = load_embedded_data(cache_dir='data/',
                                            db_file='data/identifier.sqlite',
                                            limit=1000,
                                            remove_urls=True,
                                            remove_nonals=True,
                                            remove_hashtags=True,
                                            remove_mentions=True,
                                            backend=embedding_backend,
                                            model=embedding_model)

    for metric in ['cosine', 'euclidean']:
        for n_neighbours in (2, 5, 10, 20, 50, 100, 200, 500, 1000):
            for min_dist in [0.0, 0.1, 0.3, 0.5]:
                for op_ratio in [0.3, 0.7, 1.0]:
                    tic = time.time()
                    title = f'n_neighbors={n_neighbours}, metric={metric}, min_dist={min_dist}, op_mix={op_ratio}'
                    fname = f'data/umap_params/1k/{metric}_n{n_neighbours}_dst{min_dist}_op{op_ratio}.png'
                    print(title)
                    if os.path.isfile(fname):
                        print('Skipping...')

                    fit = UMAP(
                        n_neighbors=n_neighbours,
                        min_dist=min_dist,
                        n_components=2,
                        metric=metric,
                        set_op_mix_ratio=op_ratio,
                        verbose=True
                    )

                    u = fit.fit_transform(embeddings)
                    fig = plt.figure(figsize=(10,10))
                    ax = fig.add_subplot(111)
                    ax.scatter(u[:, 0], u[:, 1])
                    tm = time.time()-tic
                    tm = f'{math.floor(tm/60)}:{tm%60}s'
                    title += f', runtime={tm}'
                    print(f'  took {tm} to execute')
                    plt.title(title, fontsize=12)

                    plt.savefig(fname)
