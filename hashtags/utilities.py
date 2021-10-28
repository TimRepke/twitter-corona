import re
from datetime import datetime
from dataclasses import dataclass
import sqlite3
from collections import defaultdict, Counter
import scipy.spatial.distance
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Union, Literal
import pandas as pd
import numpy as np


class GroupedHashtags:
    def __init__(self,
                 grouped_tweets: dict[list],
                 vectoriser: Union[TfidfVectorizer, CountVectorizer]):
        self.groups = {k: Counter([hi for twi in v for hi in twi.hashtags])
                       for k, v in grouped_tweets.items()}

        self.vectoriser = vectoriser
        self.vectors = self.vectoriser.fit_transform(self.fake_texts)
        self.vocab = {v: k for k, v in self.vectoriser.vocabulary_.items()}

    @property
    def fake_texts(self) -> list[str]:
        return [' '.join([(hashtag + ' ') * cnt for hashtag, cnt in v.items()])
                for v in self.groups.values()]

    @property
    def keys(self) -> list[str]:
        return list(self.groups.keys())

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _most_common_raw(self, top_n=5, include_count=True, include_hashtag=True, least_common=False) -> \
            list[tuple[str, Union[str, int, tuple[str, int]]]]:
        assert include_hashtag or include_count

        if least_common:
            ret = [(group, cnt.most_common()[-top_n:]) for group, cnt in self.groups.items()]
        else:
            ret = [(group, cnt.most_common(top_n)) for group, cnt in self.groups.items()]

        if include_count and include_hashtag:
            return ret
        if include_hashtag:
            return [(group, [mc[0] for mc in mcs]) for group, mcs in ret]
        if include_count:
            return [(group, [mc[0] for mc in mcs]) for group, mcs in ret]

    def _most_common_vectoriser(self, top_n=5, include_count=True, include_hashtag=True, least_common=False) -> \
            list[tuple[str, Union[str, int, float, tuple[str, int], tuple[str, float]]]]:
        assert include_hashtag or include_count

        if least_common:
            indices = np.asarray(np.argsort(self.vectors.todense(), axis=1)[:, :top_n])
        else:
            indices = np.flip(np.asarray(np.argsort(self.vectors.todense(), axis=1)[:, -top_n:]), axis=1)

        token_value_pairs = [
            [('#' + self.vocab[ind], self.vectors[row_i, ind]) for ind in row]
            for row_i, row in enumerate(indices)
        ]
        if include_count and include_hashtag:
            return [(group, tvps) for group, tvps in zip(self.groups.keys(), token_value_pairs)]
        if include_hashtag:
            return [(group, [tvp[0] for tvp in tvps]) for group, tvps in zip(self.groups.keys(), token_value_pairs)]
        if include_count:
            return [(group, [tvp[1] for tvp in tvps]) for group, tvps in zip(self.groups.keys(), token_value_pairs)]

    def most_common(self, top_n=5, from_vectoriser=False,
                    include_count=True, include_hashtag=True, least_common=False) -> \
            list[tuple[str, Union[str, int, float, tuple[str, int], tuple[str, float]]]]:
        if from_vectoriser:
            return self._most_common_vectoriser(top_n, include_count=include_count, include_hashtag=include_hashtag,
                                                least_common=least_common)
        return self._most_common_raw(top_n, include_count=include_count, include_hashtag=include_hashtag,
                                     least_common=least_common)

    def pairwise_similarities(self, metric: Literal['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                                    'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                                                    'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis',
                                                    'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
                                                    'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                                                    'wminkowski', 'yule']) -> np.ndarray:
        return scipy.spatial.distance.cdist(self.vectors.todense(), self.vectors.todense(), metric=metric)

    def get_top_hashtags_df(self, top_n=20, from_vectoriser=True, least_significant=False) -> pd.DataFrame:
        relevant_tags = self.most_common(top_n, from_vectoriser, include_count=True, include_hashtag=True,
                                         least_common=least_significant)

        return pd.DataFrame([
            {**{'group': g}, **{f'Top_{i}': f'{ht} ({cnt:.3f})' for i, (ht, cnt) in enumerate(popular)}}
            for g, popular in relevant_tags
        ])

    def get_frequencies(self, tags):
        return {tag: [grp.get(tag, 0) for grp in self.groups.values()] for tag in tags}
