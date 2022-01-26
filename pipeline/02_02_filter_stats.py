import os
from tqdm import tqdm

# DATASET = 'geoengineering'
DATASET = 'climate2'

ONLY_EN = True
ALLOW_LANG_NULL = True
MIN_TOKENS = 4
MAX_HASHTAGS = 5
FROM_DATE = '2018-01'
TO_DATE = '2022-01'

RELEVANCE_FILE = f'data/{DATASET}/tweets_relevant_{ONLY_EN}_{ALLOW_LANG_NULL}_{MIN_TOKENS}_' \
                 f'{MAX_HASHTAGS}_{FROM_DATE}_{TO_DATE}.txt'
IRRELEVANCE_FILE = f'data/{DATASET}/tweets_irrelevant_{ONLY_EN}_{ALLOW_LANG_NULL}_{MIN_TOKENS}_' \
                   f'{MAX_HASHTAGS}_{FROM_DATE}_{TO_DATE}.txt'

with open(RELEVANCE_FILE) as f_rel, open(IRRELEVANCE_FILE) as f_irrel:
    print('Reading data...')
    n_rel = sum(1 for _ in tqdm(f_rel))
    irrelevant = [[int(s) for s in l.strip().split('|')] for l in tqdm(f_irrel)]
    n_irrel = len(irrelevant)
    total = n_rel + n_irrel

    print(f'Total: {total:,}; {n_rel:,} ({n_rel / total:.2%}) relevant; '
          f'{len(irrelevant):,} ({len(irrelevant) / total:.2%}) irrelevant')

    # dup |{is_en:d}|{has_text:d}|{has_min_tokens:d}|{has_max_hashtags:d}\n')
    duplicates = sum([i[1] for i in irrelevant])
    print(f'Duplicates: {duplicates:,} ({duplicates / total:.2%})')

    nondup_irrel = [[not bool(ii) for ii in i[1:]] for i in irrelevant if i[1] == 0]
    english = sum([i[1] for i in nondup_irrel])
    print(f'non-english: {english:,} ({english / total:.2%})')
    empty = sum([i[2] for i in nondup_irrel])
    print(f'empty: {empty:,} ({empty / total:.2%})')
    short = sum([i[3] for i in nondup_irrel])
    print(f'too short: {short:,} ({short / total:.2%})')
    hashtags = sum([i[4] for i in nondup_irrel])
    print(f'too many hashtags: {hashtags:,} ({hashtags / total:.2%})')
    hs = sum([i[3] and i[4] for i in nondup_irrel])
    print(f'too short and too many hashtags: {hs:,} ({hs / total:.2%})')
    pfd = sum([not i[5] for i in nondup_irrel])
    print(f'pre from date: {pfd:,} ({pfd / total:.2%})')
    ptd = sum([i[6] for i in nondup_irrel])
    print(f'past to date: {ptd:,} ({ptd / total:.2%})')
