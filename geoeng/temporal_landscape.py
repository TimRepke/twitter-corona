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

    FORMATS = {'yearly': '%Y', 'monthly': '%Y-%m', 'weekly': '%Y-%W', 'daily': '%Y-%m-%d'}
    SELECTED_FORMAT = 'monthly'
    FORMAT = FORMATS[SELECTED_FORMAT]

    groups = {}
    for i, tw in enumerate(tweets):
        timestamp = datetime.strptime(tw['created_at'][:19], '%Y-%m-%dT%H:%M:%S')
        group = timestamp.strftime(FORMAT)
        if group not in groups:
            groups[group] = []
        groups[group].append(i)

    colormap = glasbey

    grp_keys = list(sorted(list(groups.keys())))

    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": [-70, 80], "title": ""}
    fig_dict["layout"]["yaxis"] = {"range": [-60, 80], "title": ""}  # , "type": "log"}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 300, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Month: ",
            "visible": True,
            "xanchor": "right"
        },
        # "mode": "immediate",
        # "transition": {"duration": 0},
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    FADE_STEPS = 20
    # make data
    i = FADE_STEPS
    for step in range(FADE_STEPS):
        if i < step:
            continue
        grp_key = grp_keys[i - step]
        opacity = (FADE_STEPS - step) / FADE_STEPS
        labels = topic_model.labels[groups[grp_key]]
        pts = topic_model.layout[groups[grp_key]]

        data_dict = {
            "x": pts[:, 0],
            "y": pts[:, 1],
            "mode": "markers",
            "opacity": opacity,
            "marker": {
                "color": colormap[0],
                "opacity": opacity
            }
        }
        fig_dict["data"].append(data_dict)

    # make frames
    for i, grp_key in enumerate(grp_keys):
        frame = {"data": [], "name": grp_key}
        for step in range(FADE_STEPS):
            if i < step:
                continue
            grp_key = grp_keys[i - step]
            opacity = 1 / (step + 1)
            labels = topic_model.labels[groups[grp_key]]
            pts = topic_model.layout[groups[grp_key]]

            data_dict = {
                "x": pts[:, 0],
                "y": pts[:, 1],
                "mode": "markers",
                "text": [tweets[idx]['text'] for idx in groups[grp_key]],
                "marker_color": colormap[0],
                "opacity": opacity,
                "marker": {
                    "color": colormap[0],
                    "opacity": opacity
                },
                "name": f'fade{step}'
                # "marker_color": colormap[step]
                # "text": list(dataset_by_year_and_cont["country"]),
                # "marker": {
                #     #"sizemode": "area",
                #     #"sizeref": 200000,
                #     "color": colormap[0],
                #     "size": 3# list(dataset_by_year_and_cont["pop"])
                # },
                # "name": continent
            }
            frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [grp_key],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 0}}
        ],
            "label": grp_key,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)

    fig.show()

    # STEPS_TAIL = 20
    # frames = []
    # for i in range(len(grp_keys)):
    #     frame = []
    #     for time_slot in range(STEPS_TAIL):
    #         if i < time_slot:
    #             continue
    #         grp_key = grp_keys[i - time_slot]
    #         opacity = 1 / (time_slot + 1)
    #         labels = topic_model.labels[groups[grp_key]]
    #         pts = topic_model.layout[groups[grp_key]]
    #
    #         frame.append(go.Scatter(
    #             x=pts[:, 0], y=pts[:, 1],
    #             marker_color=colormap[0],
    #             opacity=opacity,
    #             mode='markers'
    #         ))
    #     frames.append(go.Frame(data=frame))
    #
    #     # fig.update_traces(mode='markers', marker_line_width=1, marker_size=3, selector={'type': 'scatter'})
    #     # fig.update_layout(title='Styled Scatter',
    #     #                   yaxis_zeroline=False, xaxis_zeroline=False)
    #
    # fig = go.Figure(
    #     data=[go.Scatter(x=[0, 1], y=[0, 1])],
    #     layout=go.Layout(
    #         xaxis=dict(range=[-70, 80], autorange=False),
    #         yaxis=dict(range=[-60, 80], autorange=False),
    #         title="Start Title",
    #         updatemenus=[dict(
    #             type="buttons",
    #             buttons=[dict(label="Play",
    #                           method="animate",
    #                           args=[None])])]
    #     ),
    #     frames=frames
    # )
    #
    # fig.show()

    #
    fig.write_html('data/plt_topics_temporal.html')
    # fig.write_image('data/plt_emotions_static.png')
