
import json
import math
import os

from tqdm import tqdm

def count_tweets(file_path):
    print(f"Counting tweets in {file_path}...")
    with open(file_path) as f:
        num_lines = sum(1 for l in f)
        print(f"  - File contains {num_lines} tweets.")


def exit_if_exists(file_path: str):
    if os.path.exists(file_path):
        print(
            f"The file {file_path} already exists. If you are sure you want to proceed, delete it first."
        )
        exit(1)


def produce_batches(file_path: str, batch_size: int, init_skip: int=0):
    print("Counting tweets...")
    num_lines = sum(1 for l in file_path)
    print(f"  - Source file contains {num_lines} tweets.")
    n_batches = math.ceil(num_lines / batch_size)
    file_path.seek(0)

    line_num = 0
    for _ in range(init_skip):
        next(file_path)
        line_num += 1

    for batch_i in range(n_batches):
        tqdm.write(
            f"===== PROCESSING BATCH {batch_i + 1} ({(batch_i + 1) * batch_size}/{num_lines}) ====="
        )

        tweets = []
        while len(tweets) < batch_size and line_num < num_lines:
            tweets.append(json.loads(next(file_path)))
            line_num += 1

        tqdm.write(
            f'Current file pos: {line_num}; Tweets from {tweets[0]["created_at"]} to {tweets[-1]["created_at"]}'
        )
        yield tweets