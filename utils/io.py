
import json
import math
import os

from tqdm import tqdm
from typing import TextIO

def count_tweets(opened_file: TextIO):
    print(f"Counting tweets...")
    num_lines = sum(1 for _ in opened_file)
    print(f"  - File contains {num_lines} tweets.")
    return num_lines


def exit_if_exists(file_path: str):
    if os.path.exists(file_path):
        print(
            f"The file {file_path} already exists. If you are sure you want to proceed, delete it first."
        )
        exit(1)


def produce_batches(opened_file: TextIO, batch_size: int, init_skip: int=0):
    num_lines = count_tweets(opened_file)
    n_batches = math.ceil(num_lines / batch_size)
    opened_file.seek(0)

    line_num = 0
    for _ in range(init_skip):
        next(opened_file)
        line_num += 1

    for batch_i in tqdm(range(n_batches)):
        tqdm.write(
            f"===== PROCESSING BATCH {batch_i + 1} ({(batch_i + 1) * batch_size}/{num_lines}) ====="
        )

        tweets = []
        while len(tweets) < batch_size and line_num < num_lines:
            tweets.append(json.loads(next(opened_file)))
            line_num += 1

        tqdm.write(
            f'Current file pos: {line_num}; Tweets from {tweets[0]["created_at"]} to {tweets[-1]["created_at"]}'
        )
        yield tweets