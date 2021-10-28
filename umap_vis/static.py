import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils import load_embedded_data
from utils.embedding import SentenceTransformerBackend
from utils.topics import FrankenTopic, UMAPArgs, VectorizerArgs

if __name__ == '__main__':
    embedding_backend = SentenceTransformerBackend
    embedding_model = 'paraphrase-multilingual-MiniLM-L12-v2'
    tweets, embeddings = load_embedded_data(cache_dir='data/',
                                            db_file='data/identifier.sqlite',
                                            limit=50000,
                                            remove_urls=True,
                                            remove_nonals=True,
                                            remove_hashtags=True,
                                            remove_mentions=True,
                                            backend=embedding_backend,
                                            model=embedding_model)

    topic_model = FrankenTopic(
        n_topics=25,
        n_words_per_topic=15,
        n_candidates=50,
        mmr_diversity=0.7,
        vectorizer_args=VectorizerArgs(max_df=.8, stop_words='english'),
        umap_args=UMAPArgs(
            spread=1.,
            min_dist=0.05,
            densmap=False
        ),
        emb_backend=embedding_backend,
        emb_model=embedding_model
    )
    topic_model.fit(tweets, embeddings)

    tfidf_topics = topic_model.get_top_n_tfidf(20)
    mmr_topics = topic_model.get_top_n_mmr(20)
    for w1, w2 in zip(tfidf_topics, mmr_topics):
        print([w[0] for w in w1])
        print([w[0] for w in w2])
        print('--')

    # uplot.points(topic_model.umap, labels=topic_model.labels)
    # plt.show()

    fig = go.Figure()
    for topic in np.unique(topic_model.labels):
        pts = topic_model.umap[topic_model.labels == topic]
        top_words = [w[0] for w in mmr_topics[topic]][:5]
        texts = [
            tweets.tweets[idx].text
            for idx in np.argwhere(topic_model.labels == topic).reshape(-1, )
        ]
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1],
            name=f'Topic {topic+1}:<br> {", ".join(top_words)}',
            marker_color=px.colors.qualitative.Alphabet[topic],
            opacity=0.5,
            text=texts
        ))
        n_plot_words = 8
        positions = pts[np.random.choice(pts.shape[0], size=n_plot_words, replace=False)]
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
    fig.write_html('data/plt.html')
