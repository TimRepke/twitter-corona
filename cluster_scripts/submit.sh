#!/bin/bash

# load environment variables from file
export $(grep -v '^#' .env | xargs)

# submit the job
sbatch --mail-user=$MAIL_USER --workdir=$WORKDIR --job-name="${TASK}_${SOURCE_FILE}" slurm_job.sh