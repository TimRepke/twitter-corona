import math

import numpy as np
import pandas as pd
import scipy.interpolate
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import animation
from colorcet import glasbey
import sklearn.datasets
import umap
import umap.plot
import umap.utils as utils
import umap.aligned_umap
from utils import load_embedded_data
from utils.embedding import SentenceTransformerBackend
from utils.topics import FrankenTopic, UMAPArgs, VectorizerArgs, TSNEArgs, KMeansArgs, HDBSCANArgs

TILE_SIZE = 5


def axis_bounds(embedding):
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]


def plot_series(mapper_):
    N_COLS = 3
    num_rows = math.ceil(len(mapper_.embeddings_) / N_COLS)

    fig, axs = plt.subplots(num_rows, N_COLS, figsize=(TILE_SIZE * N_COLS, TILE_SIZE * num_rows))
    ax_bound = axis_bounds(np.vstack(mapper_.embeddings_))
    for i, ax in enumerate(axs.flatten()):
        if i >= len(mapper_.embeddings_):  # skip last one if uneven number
            continue
        # current_target = ordered_target[150 * i:min(ordered_target.shape[0], 150 * i + 400)]
        len_prev = 0 if i == 0 else len(mapper_.embeddings_[i - 1])
        cmap = plt.get_cmap('tab10').colors
        ax.scatter(*mapper_.embeddings_[i].T,
                   c=[cmap[0] if j < len_prev else cmap[1] for j in range(len(mapper_.embeddings_[i]))],
                   s=2, cmap="tab10")  # c=current_target,
        ax.axis(ax_bound)
        ax.set(xticks=[], yticks=[])

    plt.tight_layout()


def plot_animation(mapper_: umap.AlignedUMAP, frames_per_slice=20):
    n_slices = len(mapper_.embeddings_)
    ax_bound = axis_bounds(np.vstack(mapper_.embeddings_))

    all_frame_layouts = []

    for slice_i in range(n_slices - 1):
        slice_j = slice_i + 1
        layout_i = mapper_.embeddings_[slice_i]
        layout_j = mapper_.embeddings_[slice_j]
        mapping = mapper_.dict_relations_[slice_i]

        # mask for all the things included in the next slice
        mask = np.array([idx in mapping for idx in range(len(layout_i))], dtype=bool)

        from_layout = np.vstack([layout_i[list(mapping.keys())],
                                 layout_i[~mask]])
        to_layout = np.vstack([layout_j[list(mapping.values())],
                               layout_i[~mask]])

        z = np.linspace(0, 1.0, frames_per_slice)
        xs = scipy.interpolate.interp1d(np.linspace(0, 1.0, 2),
                                        np.vstack([from_layout[:, 0], to_layout[:, 0]]).T)(z)
        ys = scipy.interpolate.interp1d(np.linspace(0, 1.0, 2),
                                        np.vstack([from_layout[:, 1], to_layout[:, 1]]).T)(z)

        frame_layouts = np.array([xs, ys]).T
        all_frame_layouts.append(frame_layouts)

    all_frame_layouts.append(np.array([mapper_.embeddings_[-1] for _ in range(frames_per_slice)]))

    fig = plt.figure(figsize=(TILE_SIZE, TILE_SIZE), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    scat = ax.scatter([], [], s=2)
    text = ax.text(ax_bound[0] + 0.5, ax_bound[2] + 0.5, '')
    ax.axis(ax_bound)
    ax.set(xticks=[], yticks=[])
    plt.tight_layout()

    num_frames = frames_per_slice * len(all_frame_layouts)

    def animate(i):
        slice_k = i // frames_per_slice
        slice_l = i % frames_per_slice
        scat.set_offsets(all_frame_layouts[slice_k][slice_l])
        text.set_text(f'Frame {i} ({slice_k}–{slice_l})')
        return scat

    anim = animation.FuncAnimation(
        fig,
        init_func=None,
        func=animate,
        frames=num_frames,
        interval=40)
    anim.save('data/alumap.gif', writer="pillow")
    plt.close(anim._fig)


if __name__ == '__main__':
    limit = 10000

    embedding_backend = SentenceTransformerBackend
    embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'

    tweets, embeddings = load_embedded_data(cache_dir='data/',
                                            db_file='data/identifier.sqlite',
                                            limit=limit,
                                            remove_urls=True,
                                            remove_nonals=True,
                                            remove_hashtags=True,
                                            remove_mentions=True,
                                            backend=embedding_backend,
                                            model=embedding_model)

    print('Sorting...')
    # data = list(sorted(zip(tweets.tweets, embeddings), key=lambda x: x[0].date))
    data = list(zip(tweets.tweets, embeddings))
    N_SLICES = 6
    SLICE_SIZE = math.ceil(len(data) / N_SLICES)

    print('Slicing...')
    # two modes:
    #  a) data always lives on
    #  b) data decays (only some overlap between slices)
    MODE = 'a'
    if MODE == 'a':
        slices = [data[:(i + 1) * SLICE_SIZE] for i in range(N_SLICES)]
        sliced_embeddings = [np.array([e for _, e in s]) for s in slices]
    else:
        # FIXME: no overlap yet
        slices = [data[i * SLICE_SIZE:(i + 1) * SLICE_SIZE] for i in range(N_SLICES)]
        sliced_embeddings = [np.array([e for _, e in s]) for s in slices]

    print('Building relations dict')
    relation_dicts = [{i: i for i in range(len(sliced_embeddings[si]))}
                      for si in range(N_SLICES - 1)]

    print('Running UMAP...')
    alumap = umap.AlignedUMAP(verbose=True,
                              alignment_window_size=2,
                              alignment_regularisation=0.001,
                              n_neighbors=50,
                              min_dist=0.5,
                              metric='cosine')
    mapper = alumap.fit(sliced_embeddings, relations=relation_dicts)

    print('Plotting...')
    plot_series(mapper)
    plt.show()
    plt.close()

    plot_animation(mapper, frames_per_slice=30)
