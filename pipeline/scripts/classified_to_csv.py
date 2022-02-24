import json
import re
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np

models = ['cards', 'cardiff-stance-climate', 'cardiff-offensive',
          'cardiff-sentiment', 'bertweet-sentiment',
          'geomotions-orig', 'geomotions-ekman', 'cardiff-emotion', 'bertweet-emotions']

stats = {
    model: defaultdict(int)
    for model in models
}
cooc_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

print('counting')
with open('data/climate2/tweets_classified_7000000_False.jsonl', 'r') as f_in, \
        open('data/climate2/tweets_classified_7000000_False.csv', 'w') as f_out:
    f_out.write(','.join(models) + ',text\n')
    li = 0
    for line in f_in:
        li += 1
        tweet = json.loads(line)
        for model in models:
            label = list(tweet['classes'][model].keys())[0]
            stats[model][label] += 1
            f_out.write(label + ',')

            for model_cooc in models:
                if model_cooc != model:
                    label_cooc = list(tweet['classes'][model_cooc].keys())[0]
                    cooc_stats[model][label][model_cooc][label_cooc] += 1
        f_out.write(re.sub(r'(\s+|,)', ' ', tweet['text']) + '\n')
        # if li > 1000:
        #     break

# print(json.dumps(stats, indent=3))
# print(json.dumps(cooc_stats, indent=3))
print('plotting')
fig = plt.figure(figsize=(40, 40), dpi=120)
spi = 0
for i, model_i in enumerate(models):
    for j, model_j in enumerate(models):
        spi += 1
        if model_j != model_i:

            plt.subplot(len(models), len(models), spi, xmargin=10, ymargin=10)
            labels_i = sorted(list(cooc_stats[model_i].keys()), reverse=True)
            labels_j = sorted(list(cooc_stats[model_j].keys()), reverse=True)
            if model_i == 'cards':
                labels_i.remove('0_0')
            x = np.zeros((len(labels_i), len(labels_j)))
            for li, label_i in enumerate(labels_i):
                for lj, label_j in enumerate(labels_j):
                    x[li][lj] = cooc_stats[model_i][label_i][model_j][label_j]

            plt.imshow(x, interpolation='none')
            plt.ylabel(model_i, rotation=90)
            plt.xlabel(model_j)
            plt.xticks(np.arange(len(labels_j)), labels_j, rotation=90, fontsize=6)
            plt.yticks(np.arange(len(labels_i)), labels_i, fontsize=6)

print('layout+show')
fig.tight_layout()
plt.show()
