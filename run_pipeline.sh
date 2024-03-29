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

python pipeline/04_04_01_embed_remaining_tweets.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                    --cluster-ram=45G --cluster-time=4:00:00 --python-unbuffered \
                                    --file-sampled=tweets_filtered_7000000.jsonl --file-full=tweets_filtered_15000000.jsonl \
                                    --file-out=extended_tweets.npy

python pipeline/04_04_02_join_remaining_tweets_batch.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                    --cluster-ram=60G --cluster-time=8:00:00 --cluster-n-cpus=10 --python-unbuffered \
                                    --file-sampled=tweets_filtered_7000000.jsonl --file-full=tweets_filtered_15000000.jsonl \
                                    --file-emb-sample=tweets_embeddings_7000000_True_minilm.npy \
                                    --file-emb-rest=extended_tweets.npy --file-labels=topics/labels_7000000_tsne.npy \
                                    --n-neighbours=30 --target-folder=topics/full_batched/ \
                                    --metric=ip --m=20 --efc=300 --efq=100

python pipeline/04_04_03_join_landscape.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                    --cluster-ram=60G --cluster-time=8:00:00 --cluster-n-cpus=10 --python-unbuffered \
                                    --file-sampled=tweets_filtered_7000000.jsonl --file-full=tweets_filtered_15000000.jsonl \
                                    --file-emb-sample=tweets_embeddings_7000000_True_minilm.npy \
                                    --file-emb-rest=extended_tweets.npy --file-tsne-sampled=topics/layout_7000000_tsne.npy \
                                    --n-neighbours=10 --target-file=topics/tsne_full.csv --index-dir=topics/full_batched/ \
                                    --metric=ip --m=20 --efc=250 --efq=100

# ====================================================
# GeoEngineering Tweets
# ====================================================
python pipeline/03_01_embed_data.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                    --cluster-ram=45G --cluster-time=4:00:00 --python-unbuffered \
                                    --dataset=geoengineering --model=minilm \
                                    --file-in=data/geoengineering/tweets_clean.jsonl \
                                    --file-out=data/geoengineering/tweet_embeddings_minilm.npy

python pipeline/03_02_classify_data.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                       --python-unbuffered --cluster-time=1-22:00:00 --cluster-ram=55G \
                                       --batch-size=20000 --excl-hashtags --dataset=geoengineering --nrc \
                                       --file-in=data/geoengineering/tweets_clean.jsonl \
                                       --file-out=data/geoengineering/tweets_classified.jsonl \
                                       --models cardiff-sentiment cardiff-emotion cardiff-offensive \
                                                cardiff-stance-climate geomotions-orig geomotions-ekman \
                                                bertweet-sentiment bertweet-emotions

python pipeline/04_01_layout.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                --cluster-ram=60G --cluster-time=4:00:00 --python-unbuffered \
                                --tsne-n-jobs=12 --cluster-n-cpus=12 --tsne-prefit-size=150000  --tsne-neighbors=auto\
                                --tsne-prefit-perplexity=300 --tsne-early-exaggeration-iter=500 \
                                --tsne-early-exaggeration=12 --tsne-perplexity=20 --dataset=geoengineering \
                                --file-in=data/geoengineering/tweet_embeddings_minilm.npy \
                                --file-out=data/geoengineering/layout_tsne.npy

python pipeline/04_03_topicmodel.py --mode=cluster --cluster-mail=timrepke@pik-potsdam.de --cluster-user=timrepke \
                                    --cluster-n-cpus=5 --cluster-ram=40G --python-unbuffered --cluster-time=1:00:00 \
                                    --file-tweets=data/geoengineering/tweets_clean.jsonl \
                                    --file-layout=data/geoengineering/layout_tsne.npy \
                                    --file-labels=data/geoengineering/topics.npy \
                                    --output-directory=data/geoengineering/topics \
                                    --dataset=geoengineering