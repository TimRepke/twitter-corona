import json

with open('../data/geoengineering_tweets_nlu.jsonl') as f:
    for l in f:
        t = json.loads(l)
        print('-----')
        print(t['text'])

        print('  ->', ', '.join(
            [f'{k}: {t["nlu"][k][0]} ({t["nlu"][k][1]:.2f})'
             for k in ['emotions', 'sentiments', 'sentiments_twitter', 'bullying', 'spam', 'sarcasm']]))
