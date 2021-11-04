import json

with open('../data/geoengineering_tweets_tweets_emotions.jsonl') as f:
    for l in f:
        t = json.loads(l)
        print('-----')
        print(t['text'])
        cf = t['crystalfeel']
        print(f'  -> sad: {cf["sadness_scores"]}, anger: {cf["anger_scores"]}, '
              f'fear: {cf["fear_scores"]}, joy: {cf["joy_scores"]}, valence: {cf["valence_scores"]}')
