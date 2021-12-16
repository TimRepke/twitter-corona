import importlib
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

pipe_classify = importlib.import_module("pipeline.03_02_classify_data")

from typing import Any, Dict, Literal, Optional

from cluster_scripts.paths import path_like

json_like = Dict[str, Any]
model_tasks = Literal["embedding", "classification"]


DEFAULT_EMBEDDING = {
    "bertweet": {
        "model": "vinai/bertweet-large",
    }
}

DEFAULT_CLASSIFICATION = pipe_classify.MODELS


class ModelCache:
    """Class that controls the model caching process.

    Args:
        cache_dir (Path): The directory where the models should be cached to
        models_embedding (json_like): Dictionary containing the embedding models to cache
        models_classification (json_like): Dictionary containing the classification models to cache
    """

    cache_dir: Path
    models_embedding: Dict[str, Any]
    models_classification: Dict[str, Any]

    def __init__(
        self,
        cache_dir: path_like,
        models_embedding: Optional[json_like] = None,
        models_classification: Optional[json_like] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        if models_embedding is None:
            self.models_embedding = DEFAULT_EMBEDDING
        else:
            self.models_embedding = models_embedding
        if models_classification is None:
            self.models_classification = DEFAULT_CLASSIFICATION
        else:
            self.models_classification = models_classification

    def get_cache_path(self, model_name) -> Path:
        return self.cache_dir / model_name

    def is_cached(self, model_name: str) -> bool:
        return self.get_cache_path(model_name).exists()

    def cache_models(self) -> None:
        models_combined = {
            "embedding": self.models_embedding,
            "classification": self.models_classification,
        }
        for task, models_dict in models_combined.items():
            for name, info in tqdm(models_dict.items(), desc=f"Caching {task} models"):
                if not self.is_cached(name):
                    model = info["model"]
                    cache_location = self.get_cache_path(name)
                    tqdm.write(f"Caching {model}...")
                    if "labels" in info.keys():
                        tqdm.write(
                            f"Found labels associated with {name}, loading as SequenceClassification model"
                        )
                        pretrained_model = (
                            AutoModelForSequenceClassification.from_pretrained(model)
                        )
                        pretrained_model.save_pretrained(cache_location)
                    else:
                        tqdm.write(
                            f"No labels associated with {name}, assuming it to be an embedding model"
                        )
                        if "sentence-transformers" in model:
                            tqdm.write(
                                f"Seems like {name} is a SentenceTransformer model"
                            )
                            pretrained_model = SentenceTransformer(model)
                            pretrained_model.save(str(cache_location))
                        else:
                            pretrained_model = AutoModel.from_pretrained(model)
                            pretrained_model.save_pretrained(cache_location)

                    tokenizer = AutoTokenizer.from_pretrained(model)
                    tokenizer.save_pretrained(cache_location)
                    info["path"] = cache_location
                    tqdm.write(f"Cached {model} to {cache_location}")
                else:
                    tqdm.write(f"Found {name} in {self.cache_dir}")
