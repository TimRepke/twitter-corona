import hashlib
import json
import os
from typing import Optional

from tqdm import tqdm
from utils.io import count_tweets, exit_if_exists
from utils.tweets import get_hash


def is_relevant(tweet, only_en, min_tokens, max_hashtags):
    if only_en and tweet['lang'] != 'en':
        return False
    if tweet['text'] is None:
        return False
    if tweet['meta']['n_tokens_raw'] <= min_tokens and tweet['meta']['n_hashtags'] >= max_hashtags:
        return False
    return True


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
        with open(source_f, 'r') as f_in, \
                open(relevance_f, 'w') as f_rel_out, \
                open(irrelevance_f, 'w') as f_irrel_out:
            hashes = set()
            total_lines = count_tweets(source_f)
            for line_i, line in tqdm(enumerate(f_in), total=total_lines, desc="Filtering / removing duplicates"):
                tweet_o = json.loads(line)
                # print(num_lines, tweet_o)
                num_lines += 1

                lang = tweet_o.get('lang', None)
                if only_en and allow_lang_null:
                    accept_lang = lang == 'en' or lang is None
                elif only_en and not allow_lang_null:
                    accept_lang = lang == 'en'
                elif not only_en and allow_lang_null:
                    accept_lang = True
                else:  # not only_en and not allow_lang_null
                    accept_lang = lang is not None

                if from_date is not None:
                    past_from_date = tweet_o['created_at'][:len(from_date)] >= from_date
                else:
                    past_from_date = True

                if to_date is not None:
                    pre_to_date = tweet_o['created_at'][:len(to_date)] <= to_date
                else:
                    pre_to_date = True

                has_text = tweet_o['text'] is not None
                if has_text:
                    has_min_tokens = tweet_o['meta']['n_tokens_raw'] >= min_tokens
                    has_max_hashtags = tweet_o['meta']['n_hashtags'] <= max_hashtags

                # if is_relevant(tweet_o):
                if accept_lang and has_text and has_min_tokens and has_max_hashtags and past_from_date and pre_to_date:
                    h = get_hash(tweet_o, include_author=False)
                    if h not in hashes:
                        # relevant and non-duplicate
                        f_rel_out.write(f'{line_i}\n')
                        hashes.add(h)
                    else:
                        f_irrel_out.write(f'{line_i}|1|0|0|0|0|0|0\n')
                        n_duplicates += 1
                else:
                    f_irrel_out.write(f'{line_i}|0|{accept_lang:d}|{has_text:d}|{has_min_tokens:d}|'
                                      f'{has_max_hashtags:d}|{past_from_date:d}|{pre_to_date:d}\n')
                    n_irrelevant += 1
            print(f'Read {num_lines:,} lines, found {n_duplicates:,} '
                  f'duplicates and {n_irrelevant:,} irrelevant tweets.')

            # clear up memory
            del hashes

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
        limit=1000000,
        only_en=True,
        from_date='2018-01',
        to_date='2022-01',
        allow_lang_null=True,
        min_tokens=4,
        max_hashtags=5
    )
