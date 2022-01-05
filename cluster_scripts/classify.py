import importlib
import os

from cluster_scripts.download.data import upload_to_owncloud
from cluster_scripts.download.models import ModelCache
from cluster_scripts.paths import (check_for_cache_dir,
                                   get_paths_from_environment, path_like)

pipe_classify = importlib.import_module("pipeline.03_02_classify_data")


def classify_on_the_cluster(
    dataset: str,
    source_f: path_like,
    target_f: path_like,
    cache_dir: path_like,
    limit: int,
    skip_first_n_lines: int,
    batch_size: int,
):
    check_for_cache_dir(cache_dir)
    mc = ModelCache(cache_dir=cache_dir)
    pipe_classify.classify_tweets(
        dataset=dataset,
        limit=limit,
        skip_first_n_lines=skip_first_n_lines,
        batch_size=batch_size,
        source_f=source_f,
        target_f=target_f,
        models=mc.get_models_with_paths("classification"),
        use_model_cache=True
    )


if __name__ == "__main__":

    paths = get_paths_from_environment()

    classify_on_the_cluster(
        dataset=str(os.environ["DATASET"]),
        source_f=paths["source"],
        target_f=paths["classification"],
        cache_dir=paths["cache"],
        limit=int(os.environ["LIMIT_C"]),
        skip_first_n_lines=int(os.environ["SKIP_FIRST_N_LINES"]),
        batch_size=int(os.environ["BATCH_SIZE"]),
    )

    if bool(os.getenv("UPLOAD")):
        upload_to_owncloud(
            remote_path=str(paths["upload"] / paths["classification"].name),
            local_path=str(paths["classification"]),
            domain=os.getenv("OC_DOMAIN"),
            user=os.getenv("OC_USER"),
            password=os.getenv("OC_PW")
        )
