import json
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

NAME = 'cc'
START_TIME = datetime(year=2007, month=1, day=1)

counts = {}
with open(f'data/counts/{NAME}.jsonl', 'r') as f_in:
    for line in f_in:
        page = json.loads(line)
        for cnt in page['data']:
            counts[cnt['start'][:10]] = cnt['tweet_count']

time_series = []
dates = []

time_cursor = START_TIME
for day, count in sorted(counts.items(), key=lambda e: e[0]):
    while time_cursor.strftime('%Y-%m-%d') < day:
        time_series.append(0)
        dates.append(time_cursor.strftime('%Y-%m-%d'))
        time_cursor += timedelta(days=1)

    time_series.append(count)
    dates.append(time_cursor.strftime('%Y-%m-%d'))
    time_cursor += timedelta(days=1)

# for d, c in zip(dates, time_series):
#     print(f'{d}   {c:,}')

# yearly mean/min/max/std of tweets per day
yearly_agg = defaultdict(list)
for day, count in zip(dates, time_series):
    yearly_agg[day[:4]].append(count)
print('Tweets matching \'"climate change" lang:en -is:retweet -is:quote\'')
growths = []
prev = None
for yr, counts in sorted(yearly_agg.items(), key=lambda e: e[0]):
    counts = np.array(counts)
    growth = 0
    if prev is not None:
        growth = (counts.sum() - prev) / prev
        if yr >= '2011':
            growths.append(growth)
    print(f'{yr}: total = {counts.sum():,} '
          f'| perc5 = {np.percentile(counts, 5):,.0f} '
          f'| perc95 = {np.percentile(counts, 95):,.0f} '
          f'| mean = {counts.mean():,.2f} '
          f'| max = {counts.max():,} '
          f'| std = {counts.std():,.2f} '
          f'| growth = {growth:.2%}')
    prev = counts.sum()
print()
print(f'Average growth 2011-2021: {np.mean(growths):.2%}')
print('\n----------------\n')
print('Total English tweets')

yearly_agg = defaultdict(list)
with open('data/climate2/english_tweet_counts_daily_2006-2021-rt.csv', 'r') as f_in:
    next(f_in)
    for line in f_in:
        day, count = line.split(',')
        count = int(count)
        if day[:4] >= '2007':
            yearly_agg[day[:4]].append(count)

growths = []
prev = None
for yr, counts in sorted(yearly_agg.items(), key=lambda e: e[0]):
    counts = np.array(counts)
    growth = 0
    if prev is not None:
        growth = (counts.sum() - prev) / prev
        if yr >= '2011':
            growths.append(growth)
    print(f'{yr}: total = {counts.sum():,} '
          f'| perc5 = {np.percentile(counts, 5):,.0f} '
          f'| perc95 = {np.percentile(counts, 95):,.0f} '
          f'| mean = {counts.mean():,.2f} '
          f'| max = {counts.max():,} '
          f'| std = {counts.std():,.2f} '
          f'| growth = {growth:.2%}')
    prev = counts.sum()
print()
print(f'Average growth 2011-2021: {np.mean(growths):.2%}')
