import datetime
import sqlite3
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import plotly.graph_objs as go
import plotly.express as px

from utils.tweets import Tweets
from hashtags.utilities import GroupedHashtags


@st.cache(hash_funcs={sqlite3.Cursor: lambda x: None}, allow_output_mutation=True)
def load_tweets(db_file, limit: int) -> Tweets:
    return Tweets(db_file=db_file, limit=limit)


if __name__ == '__main__':
    st.title('Twitter Hashtags')

    with st.expander('Settings'):
        num_tweets = st.slider('Num Tweets', min_value=50, max_value=100000, value=100000)
        tweets = load_tweets('data/identifier.sqlite', num_tweets)

        st.subheader('Vectoriser Settings')
        vector_cols = st.columns(4)
        vectoriser_type = vector_cols[0].radio('Vectoriser Type', options=['TF-IDF', 'Count'], index=0)
        min_df = vector_cols[1].number_input('min_df', min_value=0, value=10, step=1)
        max_df = vector_cols[2].number_input('max_df', min_value=0., max_value=1., value=0.8, step=0.01)
        binary = vector_cols[3].checkbox('Binary', value=False)

        vectoriser = CountVectorizer if vectoriser_type == 'Count' else TfidfVectorizer
        vectoriser = vectoriser(min_df=min_df, max_df=max_df, binary=binary)

        st.subheader('Similarity Settings')
        col1, col2 = st.columns(2)
        grouping = col1.selectbox('Grouping', options=['%Y', '%Y-%m', '%Y-%W', '%Y-%m-%d'], index=2)
        metric = col2.selectbox('Similarity Metric', options=['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                                              'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                                                              'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                                                              'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                                                              'seuclidean', 'sokalmichener', 'sokalsneath',
                                                              'sqeuclidean', 'wminkowski', 'yule'], index=5)

    grouped_hashtags = GroupedHashtags(tweets.groupby_date(grouping), vectoriser=vectoriser)

    st.header('Data Viewer')
    col3, col4 = st.columns(2)
    from_vectoriser = col3.radio('Take scores from', options=[True, False], index=0,
                                 format_func=lambda x: 'Vectoriser' if x else 'Raw')
    least_significant = col4.radio('Head or Tail', options=[True, False], index=1,
                                   format_func=lambda x: 'Least significant' if x else 'Most significant')

    st.metric(label='Vocabulary Size', value=grouped_hashtags.vocab_size)
    df_popular_hashtags = grouped_hashtags.get_top_hashtags_df(top_n=20,
                                                               from_vectoriser=from_vectoriser,
                                                               least_significant=least_significant)
    st.dataframe(df_popular_hashtags)

    st.header('Similarity Matrix')
    with st.spinner('Computing similarity matrix...'):
        group_similarities = grouped_hashtags.pairwise_similarities(metric=metric)

    mc = grouped_hashtags.most_common(top_n=6, include_count=False, include_hashtag=True,
                                      least_common=least_significant,
                                      from_vectoriser=from_vectoriser)

    # small hack (add prefix) so plotly doesn't try to interpret this as a date format
    xy_labels = [f'd:{g}' for g in grouped_hashtags.keys]

    events = [
        (datetime.date(year=2017, month=1, day=20), 'Donald Trump'),  # Trump inauguration as president
        (datetime.date(year=2018, month=8, day=20), 'Greta'),  # Greta Thunberg started her strike
        (datetime.date(year=2019, month=9, day=1), 'Australian Bushfires'),  # out-of-control fires sprung up
        (datetime.date(year=2019, month=12, day=24), 'First Covid Case'),  # First patient in Wuhan Hospital
        (datetime.date(year=2020, month=3, day=11), 'Declared Pandemic'),  # WHO declared COVID-19 as pandemic
        (datetime.date(year=2020, month=12, day=27), 'Vaccine Rollout'),  # EU officially began vaccine rollout
    ]
    events = [{'date': dt,
               'group': dt.strftime(grouping),
               'group_idx': grouped_hashtags.keys.index(dt.strftime(grouping)),
               'text': txt}
              for dt, txt in events]

    fig = go.Figure([go.Heatmap(z=group_similarities, x=xy_labels, y=xy_labels, hoverinfo='text',
                                text=[[
                                    f"""
                                   {gi}: {', '.join(li)} <br>
                                   {gj}: {', '.join(lj)}
                                    """
                                    for gi, li in mc]
                                    for gj, lj in mc]),
                     ],
                    layout=go.Layout(
                        width=600, height=600,
                        xaxis=dict(scaleanchor='y', constrain='domain', constraintoward='center'),
                        yaxis=dict(zeroline=False, autorange='reversed', constrain='domain')
                    ))
    # fig.for_each_trace(lambda trace: trace.update(hovertext='d'))
    fig.update_yaxes(autorange=True, tickformat=grouping)
    fig.update_xaxes(tickformat=grouping)
    for event in events:
        fig.add_hline(y=event['group_idx'],
                      line_dash="dot",
                      annotation_text=event['text'],
                      annotation_position="bottom right",
                      annotation_font_size=14,
                      annotation_font_color="blue")
    st.plotly_chart(fig, use_container_width=True)

    top_tags = grouped_hashtags.most_common(top_n=1, include_count=False, include_hashtag=True,
                                            least_common=least_significant,
                                            from_vectoriser=from_vectoriser)
    st.header('Hashtag Frequencies')
    st.caption('Selection based on top tag per group.')
    top_tags = [tt[0] for g, tt in top_tags]
    tag_freqs = grouped_hashtags.get_frequencies(top_tags)
    # tag_freqs = {tag: [grp.get(tag, 0) for grp in grouped_hashtags.groups.values()] for tag in top_tags}
    tag_freqs['group'] = xy_labels
    tag_freq = pd.DataFrame(tag_freqs)
    fig = px.line(tag_freq, x='group', y=top_tags)
    st.plotly_chart(fig)

    st.header('Tweet Histogram')
    res = tweets.histogram(grouping)
    mx = max([r['freq'] for r in res])
    fig = px.bar(res, x='grp', y='freq', range_y=(0, mx + (mx * 0.02)))
    st.plotly_chart(fig)