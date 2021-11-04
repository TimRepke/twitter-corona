import nlu
import json
import math

if __name__ == '__main__':
    TOTAL = 1642400
    CHUNK_SIZE = 1000
    N_CHUNKS = math.ceil(TOTAL / CHUNK_SIZE)

    with open('data/geoengineering_tweets_tweets.jsonl') as f_in, \
        open('data/geoengineering_tweets_nlu.jsonl', 'w') as f_out:
        for chunk in range(N_CHUNKS):
            print(f'===== PROCESSING CHUNK {chunk} =====')
            tweets = [json.loads(next(f_in)) for _ in range(CHUNK_SIZE)]

            texts = [t['text'] for t in tweets]
            emotions = nlu.load('emotion').predict(texts, output_level='document')
            sentiments = nlu.load('sentiment').predict(texts, output_level='document')
            twitter_sentiments = nlu.load('en.sentiment.twitter').predict(texts, output_level='document')
            bullying = nlu.load('en.classify.cyberbullying').predict(texts, output_level='document')
            spam = nlu.load('en.classify.spam').predict(texts, output_level='document')
            sarcasm = nlu.load('en.classify.sarcasm').predict(texts, output_level='document')
            toxicity = nlu.load('en.classify.toxic').predict(texts, output_level='document')

            datasets = {
                'emotions': (emotions, 'emotion', 'emotion_confidence_confidence'),
                'sentiments': (sentiments, 'sentiment', 'sentiment_confidence'),
                'sentiments_twitter': (twitter_sentiments, 'sentiment', 'sentiment_confidence'),
                'bullying': (bullying, 'cyberbullying', 'cyberbullying_confidence_confidence'),
                'spam': (spam, 'spam', 'spam_confidence_confidence'),
                'sarcasm': (sarcasm, 'sarcasm', 'sarcasm_confidence_confidence'),
                # (toxicity) # no return? FIXME
            }
            for i, tweet in enumerate(tweets):
                tweet['nlu'] = {k: (d.iloc[i][l], d.iloc[i][s]) for k, (d, l, s) in datasets.items()}
                f_out.write(json.dumps(tweet) + '\n')
