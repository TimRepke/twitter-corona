import json
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
from utils.tweets import get_hashtags, get_urls, clean_tweet
from dataclasses import dataclass, field
import plotly.graph_objects as go
from colorcet import glasbey

MODELS = {
    'cardiff-sentiment': ['negative', 'neutral', 'positive'],
    'cardiff-emotion': ['anger', 'joy', 'optimism', 'sadness'],
    'cardiff-offensive': ['not-offensive', 'offensive'],
    'cardiff-stance-climate': ['none', 'against', 'favor'],
    'geomotions-orig': [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
        'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
        'surprise',
    ],
    'geomotions-ekman': ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'],
    'nlptown-sentiment': ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
    'bertweet-sentiment': ['negative', 'neutral', 'positive'],
    'bertweet-emotions': ['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear'],
    'bert-sst2': ['negative', 'positive']
}


def get_empty_stats() -> dict[str, dict[str, int]]:
    return {
        model: {label: 0 for label in labels}
        for model, labels in MODELS.items()
    }


@dataclass
class Group:
    n_hashtags: list[int] = field(default_factory=list)
    n_urls: list[int] = field(default_factory=list)
    n_tokens: list[int] = field(default_factory=list)
    n_chars_clean: list[int] = field(default_factory=list)
    n_chars: list[int] = field(default_factory=list)
    stats: dict[str, dict[str, int]] = field(default_factory=get_empty_stats)


def plot_stacked_area(groups_: dict[str, Group], model):
    x = list(groups_.keys())
    fig = go.Figure()
    for i, label in enumerate(MODELS[model]):
        y = [g.stats[model][label] / (sum(g.stats[model].values()) + 0.00000001) for g in groups_.values()]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            hoverinfo='x+y',
            mode='lines',
            name=label,
            line=dict(width=0.5, color=glasbey[i]),
            stackgroup='one'  # define stack group
        ))

    fig.update_layout(yaxis_range=(0, 1))
    # fig.show()
    return fig


FORMATS = {'yearly': '%Y', 'monthly': '%Y-%m', 'weekly': '%Y-%W', 'daily': '%Y-%m-%d'}
SELECTED_FORMAT = 'monthly'
FORMAT = FORMATS[SELECTED_FORMAT]

with open('data/geoengineering_tweets_sentop4.jsonl') as f:
    groups = {}

    # [ret[li.date.strftime(fmt)].append(li) for li in self.tweets]

    for line in tqdm(f):
        tweet = json.loads(line)
        timestamp = datetime.strptime(tweet['created_at'][:19], '%Y-%m-%dT%H:%M:%S')
        group = timestamp.strftime(FORMAT)
        clean_txt = clean_tweet(tweet['text'],
                                remove_hashtags=True, remove_urls=True,
                                remove_nonals=True, remove_mentions=True)

        if group not in groups:
            groups[group] = Group()

        groups[group].n_tokens.append(len(clean_txt.split(' ')))
        groups[group].n_chars_clean.append(len(clean_txt))
        groups[group].n_chars.append(len(tweet['text']))
        groups[group].n_urls.append(len(get_urls(tweet['text'])))
        groups[group].n_hashtags.append(len(get_hashtags(tweet['text'])))

        for k, v in tweet['sentiments'].items():
            groups[group].stats[k][v[0][0]] += 1

    for model in MODELS.keys():
        fig_ = plot_stacked_area(groups, model)
        fig_.write_image(f'data/emotions_fig/{SELECTED_FORMAT}_{model}.png')
    #
    # print('-----')
    # print(tweet['text'])
    #
    # titles = list(tweet['sentiments'].keys())
    # values = [f'{v[0][0]} ({v[0][1]:.2f})' for v in tweet['sentiments'].values()]
    # lengths = [max(len(t), len(v)) for t, v in zip(titles, values)]
    #
    # print(' | '.join([t.ljust(l) for t, l in zip(titles, lengths)]))
    # print(' | '.join([v.ljust(l) for v, l in zip(values, lengths)]))
    #
    # print('  ->', ', '.join(
    #     [f'{k}: {v[0][0]} ({v[0][1]:.2f})'
    #      for k, v in t['sentiments'].items()]))

example = {"id": 5435923385,
           "created_at": "2009-11-05T00:50:08Z",
           "lang": None,
           "entities": None,
           "docusercat": None,
           "note": None,
           "docownership": None,
           "author": 17165533,
           "text": "Other Controversies :: Weather Manipulation-Geoengineering...  http://ff.im/b02uW",
           "favorited": False,
           "retweeted": False,
           "truncated": False,
           "source": None,
           "source_url": None,
           "favorites_count": 0,
           "retweets_count": 0,
           "replies_count": 0,
           "in_reply_to_status": None,
           "in_reply_to_user": None,
           "retweeted_status": None,
           "place": "",
           "contributors": None,
           "coordinates": None,
           "geo": None,
           "favorites_users": None,
           "tag": None,
           "retweeted_by_user_id": [None],
           "sentiments": {
               "cardiff-sentiment": [["neutral", 0.6318885684013367], ["negative", 0.34588509798049927],
                                     ["positive", 0.022226277738809586]],
               "cardiff-emotion": [["anger", 0.7722101211547852], ["sadness", 0.18043889105319977],
                                   ["optimism", 0.035212840884923935], ["joy", 0.012138169258832932]],
               "cardiff-offensive": [["not-offensive", 0.8237721920013428], ["offensive", 0.17622782289981842]],
               "cardiff-stance-climate": [["favor", 0.6603972911834717], ["none", 0.2943183481693268],
                                          ["against", 0.04528434947133064]],
               "geomotions-orig": [["gratitude", 0.8198302984237671], ["relief", 0.07829520106315613],
                                   ["neutral", 0.07029753923416138], ["optimism", 0.008712013252079487],
                                   ["excitement", 0.00792322214692831], ["nervousness", 0.003239890094846487],
                                   ["surprise", 0.0025614467449486256], ["remorse", 0.0015383717836812139],
                                   ["approval", 0.0013035850133746862], ["grief", 0.000997062656097114],
                                   ["pride", 0.0009857284603640437], ["curiosity", 0.0006619623163715005],
                                   ["embarrassment", 0.00045413055340759456], ["fear", 0.0004239498812239617],
                                   ["disgust", 0.000418851530412212], ["admiration", 0.00037770066410303116],
                                   ["joy", 0.0003160840133205056], ["anger", 0.0002804547839332372],
                                   ["amusement", 0.0002199734590249136], ["annoyance", 0.0001896126923384145],
                                   ["desire", 0.00016772148956079036], ["disapproval", 0.00014705973444506526],
                                   ["realization", 0.00014536918024532497], ["caring", 0.00014266601647250354],
                                   ["love", 0.00012425988097675145], ["confusion", 8.992334187496454e-05],
                                   ["disappointment", 8.94167460501194e-05], ["sadness", 6.641865911660716e-05]],
               "geomotions-ekman": [["neutral", 0.8825722336769104], ["surprise", 0.06465785950422287],
                                    ["anger", 0.027140527963638306], ["sadness", 0.015949038788676262],
                                    ["joy", 0.007878749631345272], ["disgust", 0.0009275365155190229],
                                    ["fear", 0.0008741097990423441]],
               "nlptown-sentiment": [["2 stars", 0.33188772201538086], ["3 stars", 0.2920323312282562],
                                     ["1 star", 0.2567680776119232], ["4 stars", 0.0955764576792717],
                                     ["5 stars", 0.02373548038303852]]}}
