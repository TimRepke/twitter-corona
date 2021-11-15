import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from colorcet import glasbey
from collections import Counter
from datetime import datetime

from utils.tweets import clean_tweet
from utils import load_embedded_data_jsonl
from utils.embedding import SentenceTransformerBackend
from utils.topics import FrankenTopic, UMAPArgs, VectorizerArgs, TSNEArgs, KMeansArgs, HDBSCANArgs

if __name__ == '__main__':
    limit = 11000

    embedding_backend = SentenceTransformerBackend
    # embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
    embedding_model = 'vinai/bertweet-large'

    tweets, embeddings = load_embedded_data_jsonl(
        source_file='data/geoengineering_tweets_sentop4_full.jsonl',
        cache_dir='data/',
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
        cache_layout=f'data/layout_emo_{limit}.npy'
    )
    topic_model.fit([clean_tweet(t['text']) for t in tweets], embeddings)

    emo_model = 'bertweet-sentiment'
    emo_labels = ['negative', 'neutral', 'positive']
    emo_labels_heat = [('negative', 'Reds'), ('positive', 'Greens')]  # + neutral

    print(f'Using {emo_model} for sentiment/emotion annotations.')

    tfidf_topics = topic_model.get_top_n_tfidf(20)
    mmr_topics = topic_model.get_top_n_mmr(20)
    clusters, cluster_count = np.unique(topic_model.labels, return_counts=True)
    for w1, w2, c_i, c_cnt in zip(tfidf_topics, mmr_topics, clusters, cluster_count):
        print(f'>> Topic {c_i} with {c_cnt} Tweets:')
        topic_sentiments = [tweets[idx]['sentiments'][emo_model][0][0]
                            for idx in np.where(topic_model.labels == c_i)[0]]
        topic_sentiments_cnt = Counter(topic_sentiments)
        print('sentiment:', ' '.join([
            f'{l}: {topic_sentiments_cnt.get(l, 0)} '
            f'({(topic_sentiments_cnt.get(l, 0) / sum(topic_sentiments_cnt.values())) * 100:.1f}%)'
            for l in emo_labels]))
        print('tfidf:', [w[0] for w in w1])
        print('mmr:', [w[0] for w in w2])
        print('---')

    # uplot.points(topic_model.umap, labels=topic_model.labels)
    # plt.show()

    colormap = glasbey

    fig = go.Figure()
    for emo_label in emo_labels_heat:
        relevant = [i for i, t in enumerate(tweets) if t['sentiments'][emo_model][0][0] == emo_label[0]]
        pts = topic_model.layout[relevant]

        # Type: enumerated , one of ( "fill" | "heatmap" | "lines" | "none" )
        # Determines the coloring method showing the contour values. If "fill", coloring is done evenly between each
        # contour level If "heatmap", a heatmap gradient coloring is applied between each contour level. If "lines",
        # coloring is done on the contour lines. If "none", no coloring is applied on this trace.
        fig.add_trace(go.Histogram2dContour(x=pts[:, 0], y=pts[:, 1], colorscale=emo_label[1],
                                            contours={'coloring': 'heatmap', 'start': 10},
                                            opacity=0.6,
                                            showscale=False))

    for topic in np.unique(topic_model.labels):
        pts = topic_model.layout[topic_model.labels == topic]
        top_words = [w[0] for w in mmr_topics[topic]][:5]
        # top_words = [w[0] for w in tfidf_topics[topic]][:5]
        texts = [
            tweets[idx]['text']
            for idx in np.argwhere(topic_model.labels == topic).reshape(-1, )
        ]
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            name=f'Topic {topic}:<br> {", ".join(top_words)}',
            marker_color=colormap[topic],
            opacity=0.5,
            text=texts
        ))
        n_plot_words = 6
        positions = pts[np.random.choice(pts.shape[0], size=min(n_plot_words, pts.shape[0]), replace=False)]
        for tw, pos in zip(top_words[:n_plot_words], positions):
            fig.add_annotation(x=pos[0], y=pos[1], text=tw, showarrow=False,
                               font={
                                   'family': "sans-serif",
                                   'color': "#000000",
                                   'size': 14
                               })

    fig.update_traces(mode='markers', marker_line_width=1, marker_size=3, selector={'type': 'scatter'})
    fig.update_layout(title='Styled Scatter',
                      yaxis_zeroline=False, xaxis_zeroline=False)

    # fig.show()
    fig.write_html('data/plt_emotions_static.html')
    fig.write_image('data/plt_emotions_static.png')

    FORMATS = {'yearly': '%Y', 'monthly': '%Y-%m', 'weekly': '%Y-%W', 'daily': '%Y-%m-%d'}
    SELECTED_FORMAT = 'monthly'
    FORMAT = FORMATS[SELECTED_FORMAT]

    groups = {}
    for tw, to in zip(tweets, topic_model.labels):
        timestamp = datetime.strptime(tw['created_at'][:19], '%Y-%m-%dT%H:%M:%S')
        group = timestamp.strftime(FORMAT)

        if group not in groups:
            groups[group] = np.zeros_like(np.unique(topic_model.labels))

        groups[group][to] += 1

    fig = go.Figure()
    x = list(groups.keys())
    for topic in np.unique(topic_model.labels):
        top_words = [w[0] for w in mmr_topics[topic]][:5]
        y = [g[topic] / g.sum() for g in groups.values()]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            hoverinfo='x+y',
            mode='lines',
            name=f'Topic {topic}:<br> {", ".join(top_words)}',
            line={'width': 0.5, 'color': glasbey[topic]},
            stackgroup='one'  # define stack group
        ))

    fig.update_layout(yaxis_range=(0, 1))
    fig.write_html(f'data/temporal_topics_{SELECTED_FORMAT}.html')
    fig.write_image(f'data/temporal_topics_{SELECTED_FORMAT}.png')
