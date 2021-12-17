#!/bin/bash

#SBATCH --qos=short
#SBATCH --time=05:00:00

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=60000

#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-type=END,FAIL

# load the anaconda module
# module load anaconda/2020.11

# activate custom environment
# source activate $WORKDIR/$ENV_NAME

# set env variables
export PYTHONPATH=$PYTHONPATH:$HOME/twitter-corona/
export TRANSFORMERS_OFFLINE=1

# run the python script
if [ "$TASK" == "embed" ]; then
    python $HOME/twitter-corona/cluster_scripts/embed.py
elif [ "$TASK" == "classify" ]; then
    python $HOME/twitter-corona/cluster_scripts/classify.py
else; then
    echo "Task ${TASK} is not implemented yet."
fi
