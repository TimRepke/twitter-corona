import os

import owncloud
from pathlib import Path
from cluster_scripts.paths import get_paths_from_environment

def upload_to_owncloud(
    remote_path: str, local_path: str, domain: str, user: str, password: str
):
    oc = owncloud.Client(domain)
    oc.login(user_id=user, password=password)
    try:
        oc.mkdir(str(Path(remote_path).parent))
    except owncloud.owncloud.HTTPResponseError as e:
        if e.status_code != 405: # 405 is thrown if directory already exists
            raise e
    print(f"Uploading {local_path} to {domain}/{remote_path}...")
    oc.put_file(remote_path, local_path)
    print(f"Done.")

if __name__ == '__main__':

    paths = get_paths_from_environment()

    if bool(os.getenv("UPLOAD")):

        for task in ["classification", "embeddings"]:
            upload_to_owncloud(
                remote_path=str(paths['upload'] / paths[task].name),
                local_path=str(paths[task]),
                domain=os.getenv("OC_DOMAIN"),
                user=os.getenv("OC_USER"),
                password=os.getenv("OC_PW")
            )
