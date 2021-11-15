import json
import numpy as np
from matplotlib import pyplot as plt

# number of neutral tweets vs threshold on pos/neg
with open('../data/geoengineering_tweets_sentop.jsonl') as f:
    ts = [json.loads(l) for l in f]

vs = [dict(t['sentiments']['bert-sst2']) for t in ts]
vals = np.array([[v['negative'], v['positive']] for v in vs])
d = vals[:, 0] - vals[:, 1]

x = np.linspace(0, 1, num=50)

plt.plot(x, [len(d[abs(d) < threshold]) for threshold in x])
plt.grid(True)
plt.xticks(np.arange(min(x), max(x) + 0.1, 0.05), fontsize=6)
plt.show()

plt.plot(x, [sum(np.max(vals, axis=1) < th) for th in x])
plt.grid(True)
plt.xticks(np.arange(min(x), max(x) + 0.1, 0.05), fontsize=6)
plt.show()
