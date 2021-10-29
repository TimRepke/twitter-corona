import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from colorcet import glasbey

from utils import load_embedded_data
from utils.embedding import SentenceTransformerBackend
from utils.topics import FrankenTopic, UMAPArgs, VectorizerArgs, TSNEArgs, KMeansArgs, HDBSCANArgs

if __name__ == '__main__':
    limit = 100000

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

    kmeans_args = KMeansArgs(max_n_topics=50,
                             min_docs_per_topic=int((len(tweets) / 40) / 3))
    hdbscan_args = HDBSCANArgs(
        # for 1k
        # min_samples=5,
        # min_cluster_size=20,
        # for 100k
        min_samples=50,
        min_cluster_size=500,
        cluster_selection_epsilon=0.
    )
    topic_model = FrankenTopic(
        # cluster_args=hdbscan_args,
        cluster_args=kmeans_args,
        n_words_per_topic=15,
        n_candidates=200,
        mmr_diversity=0.1,
        vectorizer_args=VectorizerArgs(max_df=.8, stop_words='english'),
        dr_args=tsne_params,
        emb_backend=embedding_backend,
        emb_model=embedding_model,
        cache_layout=f'data/layout_{limit}.npy'
    )
    topic_model.fit(tweets, embeddings)

    tfidf_topics = topic_model.get_top_n_tfidf(20)
    mmr_topics = topic_model.get_top_n_mmr(20)
    clusters, cluster_count = np.unique(topic_model.labels, return_counts=True)
    for w1, w2, c_i, c_cnt in zip(tfidf_topics, mmr_topics, clusters, cluster_count):
        print(f'>> Topic {c_i} with {c_cnt} Tweets:')
        print('tfidf:', [w[0] for w in w1])
        print('mmr:', [w[0] for w in w2])
        print('---')

    # uplot.points(topic_model.umap, labels=topic_model.labels)
    # plt.show()

    colormap = glasbey

    fig = go.Figure()
    for topic in np.unique(topic_model.labels):
        pts = topic_model.layout[topic_model.labels == topic]
        top_words = [w[0] for w in mmr_topics[topic]][:5]
        texts = [
            tweets.tweets[idx].text
            for idx in np.argwhere(topic_model.labels == topic).reshape(-1, )
        ]
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            name=f'Topic {topic + 1}:<br> {", ".join(top_words)}',
            marker_color=colormap[topic],
            opacity=0.5,
            text=texts
        ))
        n_plot_words = 8
        positions = pts[np.random.choice(pts.shape[0], size=min(n_plot_words, pts.shape[0]), replace=False)]
        for tw, pos in zip(top_words[:n_plot_words], positions):
            fig.add_annotation(x=pos[0], y=pos[1], text=tw, showarrow=False,
                               font={
                                   'family': "sans-serif",
                                   'color': "#000000",
                                   'size': 14
                               })

    fig.update_traces(mode='markers', marker_line_width=1, marker_size=3)
    fig.update_layout(title='Styled Scatter',
                      yaxis_zeroline=False, xaxis_zeroline=False)
    fig.show()
    fig.write_html('data/plt_static.html')
