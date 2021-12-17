import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

path_like = Union[str, Path]


def get_paths_from_environment():
    load_dotenv(Path(__file__).parent / ".env")
    work_dir = Path(os.environ["WORKDIR"])
    dataset = os.environ["DATASET"]
    source_file = Path(os.environ["SOURCE_FILE"])
    model_cache_dir = Path(os.environ["CACHE_SUBDIR"])
    file_embeddings = os.environ["TARGET_E"]
    file_classification = os.environ["TARGET_C"]
    return {
        "work": work_dir,
        "source": work_dir / dataset / source_file,
        "embeddings": work_dir / dataset / file_embeddings,
        "classification": work_dir / dataset / file_classification,
        "cache": work_dir / model_cache_dir,
    }


def check_for_cache_dir(cache_dir: path_like):
    if not Path(cache_dir).exists():
        raise FileNotFoundError(f"Could not find model cache directory {cache_dir}")
