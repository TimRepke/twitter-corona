import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

path_like = Union[str, Path]


def get_paths_from_environment():
    load_dotenv(Path(__file__).parent / ".env")
    work_dir = Path(os.getenv("WORKDIR"))
    source_file = Path(os.getenv("SOURCE_FILE"))
    output_dir = Path(os.getenv("OUTPUT_SUBDIR"))
    model_cache_dir = Path(os.getenv("MODEL_CACHE_SUBDIR"))
    return {
        "work": work_dir,
        "source": work_dir / source_file,
        "output": work_dir / output_dir,
        "cache": work_dir / model_cache_dir,
    }
