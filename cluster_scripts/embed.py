import importlib
import os

from cluster_scripts.download.models import ModelCache

pipe_embed = importlib.import_module("pipeline.03_01_embed_data")

from cluster_scripts.paths import check_for_cache_dir, get_paths_from_environment, path_like


def embed_on_the_cluster(
    dataset: str,
    source_f: str,
    target_f: str,
    cache_dir: path_like,
    model_name: str,
    limit: int,
    include_hashtags: bool,
):
    check_for_cache_dir(cache_dir)
    mc = ModelCache(cache_dir=cache_dir)
    if mc.is_cached(model_name):
        pipe_embed.embed_tweets(
            dataset=dataset,
            source_f=source_f,
            target_f=target_f,
            model=mc.get_cache_path(model_name),
            limit=limit,
            include_hashtags=include_hashtags,
            verbose=True
        )


if __name__ == "__main__":

    paths = get_paths_from_environment()

    embed_on_the_cluster(
        dataset=str(os.getenv("DATASET")),
        source_f=paths["source"],
        target_f=paths["embeddings"],
        model_name=str(os.environ["MODEL_E"]),
        cache_dir=paths["cache"],
        limit=int(os.environ["LIMIT_E"]),
        include_hashtags=int(os.environ["INCLUDE_HASHTAGS"]),
    )
