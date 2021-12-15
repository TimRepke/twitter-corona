import hashlib
import json
import os
from typing import Optional

from tqdm import tqdm
from utils.io import exit_if_exists


def is_relevant(tweet, only_en, min_tokens, max_hashtags):
    if only_en and tweet['lang'] != 'en':
        return False
    if tweet['text'] is None:
        return False
    if tweet['meta']['n_tokens_raw'] <= min_tokens and tweet['meta']['n_hashtags'] >= max_hashtags:
        return False
    return True


def get_hash(tweet):
    return hashlib.md5(f'{tweet["author_id"]}|{tweet["clean_text"].lower()}'.encode()).digest()


def filter_dataset(
    dataset: str,
    limit: int,
    only_en: bool,
    min_tokens: int,
    max_hashtags: int,
    relevance_f: Optional[str] = None,
    irrelevance_f: Optional[str] = None,
    source_f: Optional[str] = None,
    target_f: Optional[str] = None,
):

    if source_f is None:
        source_f = f"data/{dataset}/tweets_raw.jsonl"
    if target_f is None:
        target_f = f"data/{dataset}/tweets_filtered_{limit}.jsonl"
    if relevance_f is None:
        relevance_f = f"data/{dataset}/tweets_relevant_{only_en}_{min_tokens}_{max_hashtags}.txt"
    if irrelevance_f is None:
        irrelevance_f= f'data/{dataset}/tweets_irrelevant_{only_en}_{min_tokens}_{max_hashtags}.txt'

    exit_if_exists(target_f)

    if not os.path.exists(relevance_f):
        num_lines = 0
        n_duplicates = 0
        n_irrelevant = 0
        print('Filter and remove duplicates...')
        with open(source_f, 'r') as f_in, \
                open(relevance_f, 'w') as f_rel_out, \
                open(irrelevance_f, 'w') as f_irrel_out:
            hashes = set()
            for line_i, line in tqdm(enumerate(f_in)):
                tweet_o = json.loads(line)
                num_lines += 1

                is_en = (not only_en) or (only_en and tweet_o['lang'] != 'en')
                has_text = tweet_o['text'] is not None
                has_min_tokens = tweet_o['meta']['n_tokens_raw'] >= min_tokens
                has_max_hashtags = tweet_o['meta']['n_hashtags'] <= max_hashtags

                # if is_relevant(tweet_o):
                if is_en and has_text and has_min_tokens and has_max_hashtags:
                    h = get_hash(tweet_o)
                    if h not in hashes:
                        # relevant and non-duplicate
                        f_rel_out.write(f'{line_i}\n')
                    else:
                        f_irrel_out.write(f'{line_i}|1|0|0|0|0\n')
                        n_duplicates += 1
                    hashes.add(h)
                else:
                    f_irrel_out.write(f'{line_i}|0|{is_en:d}|{has_text:d}|{has_min_tokens:d}|{has_max_hashtags:d}\n')
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
                next(f_source)
            line_rel += 1

if __name__ == '__main__':

    filter_dataset(
        dataset="climate",  # 'geoengineering'
        limit=10000,
        only_en=True,
        min_tokens=4,
        max_hashtags=5
    )
