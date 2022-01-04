import json
from typing import Optional

import numpy as np
from utils.embedding import SentenceTransformerBackend
from utils.io import exit_if_exists


def line2txt_hashtags(line):
    tweet = json.loads(line)
    return tweet["clean_text"] + (" ".join(tweet["meta"]["hashtags"]))


def line2txt_clean(line):
    tweet = json.loads(line)
    return tweet["clean_text"]


def embed_tweets(
    dataset: str,
    model: str,
    limit: int,
    include_hashtags: bool,
    verbose: bool = True,
    source_f: Optional[str] = None,
    target_f: Optional[str] = None,
):

    if source_f is None:
        source_f = f"data/{dataset}/tweets_filtered_{limit}.jsonl"
    if target_f is None:
        target_f = (
            f"data/{dataset}/tweets_embeddings_{limit}_{include_hashtags}_"
            f'{model.replace("/", "_")}.npy'
        )

    exit_if_exists(target_f)

    print("Loading texts...")
    with open(source_f) as f_in:
        if include_hashtags:
            texts = [line2txt_hashtags(l) for l in f_in]
        else:
            texts = [line2txt_clean(l) for l in f_in]

    print("Embedding texts...")
    model = SentenceTransformerBackend(model)
    embeddings = model.embed_documents(texts, verbose=verbose)

    print("Storing embeddings...")
    np.save(target_f, embeddings)


if __name__ == "__main__":

    embed_tweets(
        dataset="climate",  # 'geoengineering'
        model="vinai/bertweet-large",  # 'paraphrase-multilingual-MiniLM-L12-v2'
        limit=10000,
        include_hashtags=True,
    )
