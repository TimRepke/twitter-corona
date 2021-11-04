import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from colorcet import glasbey

from utils import load_embedded_data
from utils.embedding import SentenceTransformerBackend
from utils.topics import FrankenTopic, UMAPArgs, VectorizerArgs, TSNEArgs, KMeansArgs, HDBSCANArgs
from utils.tweets import Tweets

if __name__ == '__main__':
    limit = 100000

    print('Loading tweets...')
    tweets = Tweets(db_file='data/identifier.sqlite',
                    limit=limit,
                    remove_urls=True,
                    remove_nonals=True,
                    remove_hashtags=True,
                    remove_mentions=True)

    layout = np.load(f'data/layout_{limit}.npy')

    pattern = re.compile(r'(corona|covid)', flags=re.IGNORECASE)
    highlight = np.array([bool(pattern.match(t.clean_text)) for t in tweets.tweets], dtype=bool)
    print(sum(highlight))
    fig = go.Figure()

    texts = [t.text for t in tweets.tweets]
    fig.add_trace(go.Scatter(
        x=layout[:, 0], y=layout[:, 1],
        marker_color=['#FF0000' if h else '#cdc0d0' for h in highlight],
        opacity=0.8,
        text=texts
    ))

    fig.update_traces(mode='markers', marker_line_width=0, marker_size=4)
    fig.update_layout(title='Styled Scatter',
                      yaxis_zeroline=False, xaxis_zeroline=False)
    fig.show()
    fig.write_html('data/plt_static_corona.html')
