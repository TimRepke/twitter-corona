from cluster_scripts.download.data import download_from_owncloud
from cluster_scripts.download.models import ModelCache
from cluster_scripts.paths import get_paths_from_environment
import os


if __name__ == "__main__":

    paths = get_paths_from_environment()
    
    # download the tweets as a jsonl file
    download_from_owncloud(
        remote_path=os.getenv("OC_FILE_PATH"),
        local_path=paths["source"],
        domain=os.getenv("OC_DOMAIN"),
        user=os.getenv("OC_USER"),
        password=os.getenv("OC_PW"),
    )

    # define the models to use
    model_cache = ModelCache(cache_dir=paths["cache"])
    model_cache.cache_models()

    
    



