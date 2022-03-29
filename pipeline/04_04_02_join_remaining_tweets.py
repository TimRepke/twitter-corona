import json
import os
from copy import deepcopy
from typing import Optional, Union
from tqdm import tqdm
import hnswlib
import numpy as np
from scipy.spatial.distance import cosine

from utils.cluster import ClusterJobBaseArguments
from utils.io import exit_if_exists

PathLike = Union[str, bytes, os.PathLike]


class TweetExtendArgs(ClusterJobBaseArguments):
    file_sampled: Optional[str] = None  # The file containing all already processed (down-sampled) tweets
    file_full: Optional[str] = None  # The file containing all relevant tweets
    file_emb_sample: Optional[str] = None
    file_emb_rest: Optional[str] = None
    file_labels: Optional[str] = None

    target_folder: Optional[str] = None  # The path to write outputs to

    dataset: Optional[str] = 'climate2'  # Name of the dataset

    n_neighbours: int = 20

    cluster_jobname: str = 'twitter-expand-join'
    cluster_workdir: str = 'twitter'


class MajorityIndex:
    def __init__(self, labels: np.ndarray, embeddings: np.ndarray, n_neighbours: int, n_threads: int):
        self.labels = labels
        self.n_neighbours = n_neighbours
        self.n_threads = n_threads
        ids = np.arange(len(labels))

        self.index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
        self.index.init_index(max_elements=len(labels), ef_construction=200, M=16)
        self.index.add_items(embeddings, ids, num_threads=self.n_threads)
        self.index.set_ef(1.5 * n_neighbours)  # ef should always be > k

    def get_neighbours(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.index.knn_query(embeddings, k=self.n_neighbours, num_threads=self.n_threads)

    def get_majority_labels(self, embeddings: np.ndarray) -> np.ndarray:
        ids, _ = self.get_neighbours(embeddings)
        majority_labels = []
        for ids_ in ids:
            labels = self.labels[ids_]
            unique_labels, counts = np.unique(labels, return_counts=True)
            majority_labels.append(unique_labels[counts.argmax()])
        return np.array(majority_labels)

    def get_majority_label(self, embedding: np.ndarray) -> np.ndarray:
        return self.get_majority_labels(np.array([embedding]))[0]


class ProximityIndex:
    def __init__(self, labels: np.ndarray, embeddings: np.ndarray):
        self.topics = np.unique(labels)[1:]  # with this method, outliers are not allowed
        self.centroids = np.array([np.mean(embeddings[labels == topic_i], axis=0) for topic_i in self.topics])

    def get_closest(self, embedding: np.ndarray):
        distances = np.array([cosine(embedding, centroid) for centroid in self.centroids])
        return self.topics[distances.argmax()]


def read_known_tweet_ids(source_sampled: PathLike):
    with open(source_sampled, 'r') as f_sampled:
        return [int(json.loads(line)['id']) for line in f_sampled]


def assign_topics(file_sampled: PathLike,
                  file_full: PathLike,
                  file_emb_sample: PathLike,
                  file_emb_rest: PathLike,
                  file_labels: PathLike,
                  target_folder: str,
                  n_neighbours: int,
                  n_threads: int):
    exit_if_exists(target_folder)

    print('Reading ids of already assigned tweets...')
    existing_ids = np.array(read_known_tweet_ids(file_sampled))
    existing_ids_map = {idx: i for i, idx in enumerate(existing_ids)}
    print('Reading labels of already assigned tweets...')
    existing_labels = np.load(file_labels)

    print('Loading existing embeddings...')
    existing_embeddings = np.load(file_emb_sample)
    print('Loading ids and embeddings of unassigned tweets')
    with open(file_emb_rest, 'rb') as f:
        new_ids = np.load(f)
        new_embeddings = np.load(f)
    new_ids_map = {idx: i for i, idx in enumerate(new_ids)}

    print('Building majority vote index...')
    majority_index = MajorityIndex(labels=existing_labels, embeddings=existing_embeddings,
                                   n_neighbours=n_neighbours, n_threads=n_threads)
    print('Building closest centroid index...')
    proximity_index = ProximityIndex(labels=existing_labels, embeddings=existing_embeddings)

    print('Assigning topic labels to tweets...')
    # labels (for each strategy) where *all* tweets are re-assigned to a topic
    labels_fresh_majority = []
    labels_fresh_proximity = []
    # labels (for each strategy) where only new tweets are assigned to a topic
    labels_keep_majority = []
    labels_keep_proximity = []
    with open(file_full, 'r') as f:
        for line in tqdm(f):
            tweet = json.loads(line)
            tweet_id = int(tweet['id'])

            if tweet_id in existing_ids_map:
                embedding = existing_embeddings[existing_ids_map[tweet_id]]
            else:
                embedding = new_embeddings[new_ids_map[tweet_id]]

            label_majority = majority_index.get_majority_label(embedding)
            label_proximity = proximity_index.get_closest(embedding)

            labels_fresh_majority.append(label_majority)
            labels_fresh_proximity.append(label_proximity)

            if tweet_id in existing_ids_map:
                labels_keep_majority.append(existing_labels[existing_ids_map[tweet_id]])
                labels_keep_proximity.append(existing_labels[existing_ids_map[tweet_id]])
            else:
                labels_keep_majority.append(label_majority)
                labels_keep_proximity.append(label_proximity)

    print('Saving...')
    os.makedirs(target_folder, exist_ok=True)
    np.save(os.path.join(target_folder, 'labels_fresh_majority.npy'), np.array(labels_fresh_majority))
    np.save(os.path.join(target_folder, 'labels_fresh_proximity.npy'), np.array(labels_fresh_proximity))
    np.save(os.path.join(target_folder, 'labels_keep_majority.npy'), np.array(labels_keep_majority))
    np.save(os.path.join(target_folder, 'labels_keep_proximity.npy'), np.array(labels_keep_proximity))


if __name__ == '__main__':
    args = TweetExtendArgs(underscores_to_dashes=True).parse_args()
    if args.args_file is not None:
        print(f'Dropping keyword arguments and loading from file: {args.args_file}')
        args = TweetExtendArgs().load(args.args_file)

    if args.mode == 'cluster':
        from utils.cluster import Config as SlurmConfig
        from utils.cluster.job import ClusterJob
        from utils.cluster.files import FileHandler

        s_config = SlurmConfig.from_args(args,
                                         env_vars_run={
                                             'OPENBLAS_NUM_THREADS': 1,
                                             'TRANSFORMERS_OFFLINE': 1
                                         })
        file_handler = FileHandler(config=s_config,
                                   local_basepath=os.getcwd(),
                                   requirements_txt='requirements_cluster.txt',
                                   include_dirs=['pipeline', 'utils'])
        s_job = ClusterJob(config=s_config, file_handler=file_handler)
        cluster_args = deepcopy(args)

        cluster_args.file_sampled = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_sampled}')
        cluster_args.file_full = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_full}')
        cluster_args.file_emb_sample = os.path.join(s_config.datadir_path,
                                                    f'data/{args.dataset}/{args.file_emb_sample}')
        cluster_args.file_emb_rest = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_emb_rest}')
        cluster_args.file_labels = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.file_labels}')
        cluster_args.target_folder = os.path.join(s_config.datadir_path, f'data/{args.dataset}/{args.target_folder}')

        cluster_args.model_cache = s_config.modeldir_path
        s_job.submit_job(main_script='pipeline/04_04_02_join_remaining_tweets.py', params=cluster_args)
    else:
        assign_topics(
            file_sampled=args.file_sampled,
            file_full=args.file_full,
            file_labels=args.file_labels,
            file_emb_rest=args.file_emb_rest,
            file_emb_sample=args.file_emb_sample,
            target_folder=args.target_folder,
            n_threads=args.cluster_n_cpus,
            n_neighbours=args.n_neighbours
        )
