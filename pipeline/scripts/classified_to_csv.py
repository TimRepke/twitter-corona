import json
import re
from collections import defaultdict

models = ['cards', 'cardiff-sentiment', 'cardiff-emotion', 'cardiff-offensive',
          'cardiff-stance-climate', 'geomotions-orig', 'geomotions-ekman',
          'bertweet-sentiment', 'bertweet-emotions']

stats = {
    model: defaultdict(int)
    for model in models
}
with open('data/climate2/tweets_classified_7000000_False.jsonl', 'r') as f_in, \
        open('data/climate2/tweets_classified_7000000_False.csv', 'w') as f_out:
    f_out.write(','.join(models) + ',text\n')

    for line in f_in:
        tweet = json.loads(line)
        for model in models:
            label = list(tweet['classes'][model].keys())[0]
            stats[model][label] += 1
            f_out.write(label + ',')
        f_out.write(re.sub(r'(\s+|,)', ' ', tweet['text']) + '\n')

print(json.dumps(stats, indent=3))
