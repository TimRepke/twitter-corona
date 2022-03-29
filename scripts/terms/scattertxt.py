import scattertext as st
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

with open('data/climate2/tweets_filtered_10000.jsonl') as f:
    tweets = [json.loads(l) for l in f]
lim = 10000
# texts = [t['text'] for t in tweets[:1000]]
texts = [t['clean_text'] for t in tweets[:lim]]
targets = ['pre' if t['created_at'][:4] < '2020' else 'post' for t in tweets[:lim]]
meta = [t['created_at'][:7] for t in tweets[:lim]]

data = pd.DataFrame({'text': texts, 'cat': targets, 'month': meta})

df = data.assign(
    parse=lambda df: df.text.apply(st.whitespace_nlp_with_sentences)
)

corpus = st.CorpusFromParsedDocuments(df, category_col='cat', parsed_col='parse'
).build().get_unigram_corpus().compact(st.AssociationCompactor(2000))

html = st.produce_scattertext_explorer(
    corpus,
    category='pre', category_name='pre pandemic', not_category_name='post pandemic',
    minimum_term_frequency=0, pmi_threshold_coefficient=0,
    width_in_pixels=1000, metadata=corpus.get_df()['month'],
    transform=st.Scalers.dense_rank
)
open('data/climate2/topics_big2/demo_compact.html', 'w').write(html)
