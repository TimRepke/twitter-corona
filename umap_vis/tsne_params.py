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
import openTSNE

if __name__ == '__main__':
    embedding_backend = SentenceTransformerBackend
    embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
    tweets, embeddings = load_embedded_data(cache_dir='data/',
                                            db_file='data/identifier.sqlite',
                                            limit=10000,
                                            remove_urls=True,
                                            remove_nonals=True,
                                            remove_hashtags=True,
                                            remove_mentions=True,
                                            backend=embedding_backend,
                                            model=embedding_model)

    for metric in ['cosine', 'euclidean']:
        for dof in [0.5, 1.]:
            for perp in [10, 30, 500]:
                title = f'perplexity={perp}, metric={metric}, dof={dof}'
                fname = f'data/tsne_params/10k/{metric}_p{perp}_dof{dof}.png'

                print(title)
                if os.path.isfile(fname):
                    print('Skipping...')

                tic = time.time()
                u = openTSNE.TSNE(
                    perplexity=perp,
                    initialization="pca",
                    metric=metric,
                    n_jobs=8,
                    dof=dof,
                    random_state=3
                ).fit(embeddings)

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(u[:, 0], u[:, 1])
                tm = time.time() - tic
                tm = f'{math.floor(tm / 60)}:{tm % 60:.2f}s'
                title += f', runtime={tm}'
                print(f'  took {tm} to execute')
                plt.title(title, fontsize=12)

                plt.savefig(fname)
