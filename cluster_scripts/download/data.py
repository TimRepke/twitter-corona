from pathlib import Path

import owncloud


def download_from_owncloud(
    remote_path: str, local_path: str, domain: str, user: str, password: str
):
    oc = owncloud.Client(domain)
    oc.login(user_id=user, password=password)
    output_path = Path(local_path)
    if output_path.exists():
        question = f"{output_path} already exists, overwrite (y/n)? "
        response = input(question)
        while response not in ["y", "n"]:
            response = input(question)
        if response == "y":
            output_path.unlink()
        else:
            return

    if not output_path.exists():
        output_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"Downloading {remote_path} from {domain}...")
    oc.get_file(remote_path=remote_path, local_file=str(output_path))
    print(f"Downloaded {remote_path} to {local_path}")


def upload_to_owncloud(
    remote_path: str, local_path: str, domain: str, user: str, password: str
):
    oc = owncloud.Client(domain)
    oc.login(user_id=user, password=password)
    oc.mkdir(str(Path(remote_path).parent))
    print(f"Uploading {local_path} to {domain}/{remote_path}...")
    oc.put_file(remote_path, local_path)
    print(f"Done.")