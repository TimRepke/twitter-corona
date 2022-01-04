import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, List

import numpy as np
from adaptnlp import EasySequenceClassifier
from tqdm import tqdm
from utils.io import exit_if_exists, produce_batches
from pipeline.models.classification import MODELS

# https://github.com/dhs-gov/sentop/


def assess_tweets(texts: List[str], model, labels):
    def process_result(scores):
        srt = np.argsort(scores)
        # return [f'{labels[i]} ({scores[i]:.2f})' for i in reversed(srt)]
        scores_lst = scores.tolist()
        return [(labels[i], scores_lst[i]) for i in reversed(srt)]

    classifier = EasySequenceClassifier()
    res = classifier.tag_text(text=texts, model_name_or_path=model)
    return [process_result(r) for r in res["probs"]]


def classify_tweets(
    dataset: str,
    limit: int,
    skip_first_n_lines: int,
    batch_size: int,
    source_f: Optional[str] = None,
    target_f: Optional[str] = None,
    use_model_cache: bool = False,
    models: Optional[Dict[str, Any]] = None
):

    if source_f is None:
        source_f = f"data/{dataset}/tweets_filtered_{limit}.jsonl"
    if target_f is None:
        target_f = f"data/{dataset}/tweets_sentiment_{limit}.jsonl"
    if models is None:
        models = MODELS

    exit_if_exists(target_f)

    with open(source_f) as f_in, open(target_f, "w") as f_out:
        for tweets_batch in tqdm(produce_batches(f_in, batch_size, skip_first_n_lines)):
            texts = [t["clean_text"] for t in tweets_batch]

            results = {}
            for model_name, info in models.items():
                start = time.time()
                tqdm.write(f"[{datetime.now()}] Applying model {model_name}...")
                key_to_use = "path" if use_model_cache else "model"
                results[model_name] = assess_tweets(
                    texts, model=info[key_to_use], labels=info["labels"]
                )
                secs = time.time() - start
                tqdm.write(f"  - Done after {secs // 60:.0f}m {secs % 60:.0f}s")

            for i, tweet in enumerate(tweets_batch):
                tweet["sentiments"] = {m: results[m][i] for m in MODELS.keys()}
                f_out.write(json.dumps(tweet) + "\n")


if __name__ == "__main__":

    classify_tweets(
        dataset="climate",  # 'geoengineering'
        limit=10000,
        skip_first_n_lines=0,
        batch_size=1500,
    )