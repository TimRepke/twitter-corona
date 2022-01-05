import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from torch import cuda
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines import TextClassificationPipeline
from utils.io import exit_if_exists, produce_batches

from pipeline.models.classification import MODELS

# https://github.com/dhs-gov/sentop/


def assess_tweets(texts: List[str], model, labels):
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
        model, 
        num_labels=len(labels),
        label2id={k: i for i, k in enumerate(labels)},
        id2label={i: k for i, k in enumerate(labels)})
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    device = 0 if cuda.is_available() else -1
    classifier = TextClassificationPipeline(
        model=pretrained_model, tokenizer=tokenizer, device=device)
    output = classifier(texts)
    return [(o['label'], o['score']) for o in output]


def classify_tweets(
    dataset: str,
    limit: int,
    skip_first_n_lines: int,
    batch_size: int,
    use_model_cache: bool = False,
    source_f: Optional[str] = None,
    target_f: Optional[str] = None,
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
