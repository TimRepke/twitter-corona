import hashlib
import json
import os
from typing import Optional

from tqdm import tqdm
from utils.io import count_tweets, exit_if_exists
from utils.tweets import TweetFilter, FilterResult


def filter_dataset(dataset: str,
                   limit: int,
                   only_en: bool,
                   allow_lang_null: bool,
                   min_tokens: int,
                   max_hashtags: int,
                   from_date: str = None,
                   to_date: str = None,
                   relevance_f: Optional[str] = None,
                   irrelevance_f: Optional[str] = None,
                   source_f: Optional[str] = None,
                   target_f: Optional[str] = None):
    if source_f is None:
        source_f = f'data/{dataset}/tweets_clean.jsonl'
    if target_f is None:
        target_f = f'data/{dataset}/tweets_filtered_{limit}.jsonl'
    if relevance_f is None:
        relevance_f = f'data/{dataset}/tweets_relevant_{only_en}_{allow_lang_null}_{min_tokens}_' \
                      f'{max_hashtags}_{from_date}_{to_date}.txt'
    if irrelevance_f is None:
        irrelevance_f = f'data/{dataset}/tweets_irrelevant_{only_en}_{allow_lang_null}_{min_tokens}_' \
                        f'{max_hashtags}_{from_date}_{to_date}.txt'

    exit_if_exists(target_f)

    if not os.path.exists(relevance_f):
        num_lines = 0
        n_duplicates = 0
        n_irrelevant = 0

        tweet_filter = TweetFilter(only_en=only_en, allow_lang_null=allow_lang_null, min_tokens=min_tokens,
                                   allow_duplicates=False, max_hashtags=max_hashtags, from_date=from_date,
                                   to_date=to_date, duplicate_include_author=False)

        with open(source_f, 'r') as f_in, \
                open(relevance_f, 'w') as f_rel_out, \
                open(irrelevance_f, 'w') as f_irrel_out:

            for line_i, line in tqdm(enumerate(f_in)):
                tweet_o = json.loads(line)
                num_lines += 1

                relevance = tweet_filter.is_relevant(tweet_o)
                if relevance.accept:
                    f_rel_out.write(f'{line_i}\n')
                else:
                    f_irrel_out.write(f'{line_i}|{relevance.duplicate:d}|{relevance.accept_lang:d}|'
                                      f'{relevance.has_text:d}|{relevance.has_min_tokens:d}|'
                                      f'{relevance.has_max_hashtags:d}|{relevance.past_from_date:d}|'
                                      f'{relevance.pre_to_date:d}\n')
                    if relevance.duplicate:
                        n_duplicates += 1
                    else:
                        n_irrelevant += 1

            print(f'Read {num_lines:,} lines, found {n_duplicates:,} '
                  f'duplicates and {n_irrelevant:,} irrelevant tweets.')

        # clear up memory
        del tweet_filter

        N_LINES = num_lines - n_duplicates - n_irrelevant
    else:
        print(f'I\'m using the already existing relevance file {relevance_f}!')
        with open(relevance_f) as f:
            N_LINES = sum(1 for _ in f)

    SKIP_LINES = max(int(N_LINES / limit), 1)

    print(f'Aiming to reduce size of the dataset from {N_LINES:,} to {limit:,} '
          f'by skipping {SKIP_LINES} relevant non-duplicate tweets.')
    with open(source_f) as f_source, open(relevance_f) as f_rel, open(target_f, 'w') as f_out:
        line_source = 0
        line_rel = 0

        while line_rel < N_LINES:
            if (line_rel % SKIP_LINES) == 0:
                next_source_line = int(next(f_rel))
                line = None
                while line_source <= next_source_line:
                    line = next(f_source)
                    line_source += 1
                if line is None:
                    break
                f_out.write(line)
            else:  # skip / ignore line
                next(f_rel)
            line_rel += 1


if __name__ == '__main__':
    filter_dataset(
        dataset='climate2',  # 'geoengineering'
        limit=7000000,
        only_en=True,
        from_date='2018-01',
        to_date='2022-01',
        allow_lang_null=True,
        min_tokens=4,
        max_hashtags=5
    )
