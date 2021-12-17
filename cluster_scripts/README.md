# Tweet analysis on the PIK cluster

The scripts expect some environment variables to be set. These are loaded from a `.env` file
which can be placed in the `cluster_scripts` directory. It should contain the following variables:

```
# Owncloud
OC_DOMAIN=https://mycloud.mcc-berlin.net
OC_USER=your-username
OC_PW=your-password
OC_FILE_PATH=path-to-jsonl-file-to-download

# SLURM
MAIL_USER=email-for-slurm-notifications
WORKDIR=/p/tmp/your-username/your-workdir
ENV_NAME=your-env-name

# Common
DATASET=your_dataset
SOURCE_FILE=tweets_clean_filtered.jsonl
CACHE_SUBDIR=model_cache

# Task selection
TASK=embed # must be either 'embed' or 'classify'

# Embedding
MODEL_E=bertweet
TARGET_E=embeddings.jsonl
LIMIT_E=1000
INCLUDE_HASHTAGS=False

# Classification
TARGET_C=sentiment.jsonl
LIMIT_C=1000
CHUNK_SIZE=100
SKIP_FIRST_N_LINES=0
BATCH_SIZE=500
```