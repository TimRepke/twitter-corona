#!/bin/bash

# load environment variables from file
export $(grep -v '^#' .env | xargs)

# load the anaconda module
module load anaconda/2020.11

# activate the environment
source activate $WORKDIR/$ENV_NAME

# upload the results
export PYTHONPATH=$PYTHONPATH:$HOME/twitter-corona/
python $HOME/twitter-corona/cluster_scripts/upload.py
