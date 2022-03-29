#!/bin/bash

export PYTHONPATH=$HOME/workspace/twitter-corona

# upload data by hand or add --upload-data to the first job!
# also remember, you may have to run the first job with --init-cluster

python pipeline/03_01_embed_data.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                    --cluster-ram=45G --limit=7000000 --cluster-time=4:00:00 --python-unbuffered

python pipeline/04_01_layout.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                --cluster-ram=100G --limit=7000000 --cluster-time=6:00:00 --python-unbuffered \
                                --tsne-n-jobs=30 --cluster-n-cpus=30 --tsne-prefit-size=150000 \
                                --tsne-prefit-perplexity=300 --tsne-early-exaggeration-iter=500 \
                                --tsne-early-exaggeration=12 --tsne-perplexity=20

python pipeline/04_02_hdbscan_param_sweep.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de \
                                             --cluster-user=timrepke --cluster-n-cpus=32 --cluster-ram=30G \
                                             --python-unbuffered --cluster-time=2:00:00 --limit=7000000

python pipeline/04_03_topicmodel.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                    --cluster-n-cpus=32 --cluster-ram=80G --python-unbuffered --cluster-time=4:00:00 \
                                    --limit=7000000 --cluster-selection-epsilon=0.009 --min-samples=10 --min-cluster-size=200

python pipeline/03_02_classify_data.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                       --python-unbuffered --cluster-time=10:00:00 --cluster-ram=40G --limit=7000000 \
                                       --batch-size=10000 --excl-hashtags
