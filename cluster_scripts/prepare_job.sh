#!/bin/bash

# load environment variables from file
export $(grep -v '^#' .env | xargs)

# load the anaconda module
module load anaconda/2020.11

# create the working directory
mkdir -p $WORKDIR

# create anaconda environment on the tmp partition
conda create --prefix=$WORKDIR/$ENV_NAME -y python=3.8

# activate the environment
source activate $WORKDIR/$ENV_NAME

# install dependencies
pip install \
    --no-input \
    -r requirements_cluster.txt

# download data & pretrained model
export PYTHONPATH=$PYTHONPATH:$HOME/twitter-corona/
export OPENBLAS_NUM_THREADS=1
python $HOME/twitter-corona/cluster_scripts/prepare.py
