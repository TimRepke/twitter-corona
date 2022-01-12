import os

from cluster_scripts.download.data import upload_to_owncloud
from cluster_scripts.paths import get_paths_from_environment

if __name__ == '__main__':

    paths = get_paths_from_environment()

    if bool(os.getenv("UPLOAD")):
        upload_to_owncloud(
            remote_path=str(paths["upload"] / paths["classification"].name),
            local_path=str(paths["classification"]),
            domain=os.getenv("OC_DOMAIN"),
            user=os.getenv("OC_USER"),
            password=os.getenv("OC_PW")
        )

        upload_to_owncloud(
            remote_path=str(paths["upload"] / paths["embeddings"].name),
            local_path=str(paths["embeddings"]),
            domain=os.getenv("OC_DOMAIN"),
            user=os.getenv("OC_USER"),
            password=os.getenv("OC_PW")
        )