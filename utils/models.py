from pathlib import Path
import re
import os
from sentence_transformers import SentenceTransformer
from transformers import (AutoModel,
                          AutoModelForSequenceClassification,
                          AutoTokenizer, TextClassificationPipeline)
import torch
from typing import Literal, Union, List
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class Classifier:
    hf_name: str
    labels: List[str]


@dataclass
class Embedder:
    hf_name: str
    kind: Literal['transformer', 'auto']


# to find more models, browse this page:
# https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
# Hint: the search function doesn't really work...
CLASSIFIERS = {
    # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/sentiment/mapping.txt
    'cardiff-sentiment': Classifier(hf_name='cardiffnlp/twitter-roberta-base-sentiment',
                                    labels=['negative', 'neutral', 'positive']),

    # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/emotion/mapping.txt
    'cardiff-emotion': Classifier(hf_name='cardiffnlp/twitter-roberta-base-emotion',
                                  labels=['anger', 'joy', 'optimism', 'sadness']),

    # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/offensive/mapping.txt
    'cardiff-offensive': Classifier(hf_name='cardiffnlp/twitter-roberta-base-offensive',
                                    labels=['not-offensive', 'offensive']),

    # https://github.com/cardiffnlp/tweeteval/blob/main/datasets/stance/mapping.txt
    'cardiff-stance-climate': Classifier(hf_name='cardiffnlp/twitter-roberta-base-stance-climate',
                                         labels=['none', 'against', 'favor']),

    # https://huggingface.co/monologg/bert-base-cased-goemotions-original/blob/main/config.json
    'geomotions-orig': Classifier(hf_name='monologg/bert-base-cased-goemotions-original',
                                  labels=[
                                      'admiration',
                                      'amusement',
                                      'anger',
                                      'annoyance',
                                      'approval',
                                      'caring',
                                      'confusion',
                                      'curiosity',
                                      'desire',
                                      'disappointment',
                                      'disapproval',
                                      'disgust',
                                      'embarrassment',
                                      'excitement',
                                      'fear',
                                      'gratitude',
                                      'grief',
                                      'joy',
                                      'love',
                                      'nervousness',
                                      'neutral',
                                      'optimism',
                                      'pride',
                                      'realization',
                                      'relief',
                                      'remorse',
                                      'sadness',
                                      'surprise',
                                  ]),

    # https://huggingface.co/monologg/bert-base-cased-goemotions-ekman/blob/main/config.json
    'geomotions-ekman': Classifier(hf_name='monologg/bert-base-cased-goemotions-ekman',
                                   labels=['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']),

    # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment/blob/main/config.json
    'nlptown-sentiment': Classifier(hf_name='nlptown/bert-base-multilingual-uncased-sentiment',
                                    labels=['1 star', '2 stars', '3 stars', '4 stars', '5 stars']),

    # https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis
    'bertweet-sentiment': Classifier(hf_name='finiteautomata/bertweet-base-sentiment-analysis',
                                     labels=['negative', 'neutral', 'positive']),

    # https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis
    'bertweet-emotions': Classifier(hf_name='finiteautomata/bertweet-base-emotion-analysis',
                                    labels=['others', 'joy', 'sadness', 'anger', 'surprise', 'disgust', 'fear']),
    # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/blob/main/config.json
    'bert-sst2': Classifier(hf_name='distilbert-base-uncased-finetuned-sst-2-english',
                            labels=['negative', 'positive']),
}
EMBEDDERS = {
    'bertweet': Embedder(hf_name='vinai/bertweet-large', kind='transformer'),
    'minilm': Embedder(hf_name='paraphrase-multilingual-MiniLM-L12-v2', kind='transformer'),

}


class BaseEmbedder(ABC):
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model

    @abstractmethod
    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        raise NotImplementedError()

    def embed_words(self,
                    words: List[str],
                    verbose: bool = False) -> np.ndarray:
        return self.embed(words, verbose)

    def embed_documents(self,
                        documents: List[str],
                        verbose: bool = False) -> np.ndarray:
        return self.embed(documents, verbose)


class AutoModelBackend(BaseEmbedder):
    def __init__(self, embedding_model: str, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model, use_fast=False)
        self.embedding_model = AutoModel.from_pretrained(embedding_model)

    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        tokenised = torch.tensor([self.tokenizer.encode(d) for d in documents])
        embeddings = self.embedding_model.encode(tokenised, show_progress_bar=verbose)
        return embeddings


class SentenceTransformerBackend(BaseEmbedder):
    def __init__(self, embedding_model: str, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(embedding_model)

    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings


class ModelCache:
    """Class that controls the model caching process.

    Args:
        cache_dir (Path): The directory where the models should be cached to
    """

    cache_dir: Path

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir)

    def get_model_path(self, model_name: str) -> Path:
        return self.cache_dir / self.to_safe_name(model_name)

    def is_cached(self, model_name: str) -> bool:
        return self.get_model_path(model_name).exists()

    @staticmethod
    def to_safe_name(name: str):
        return re.sub(r'[^A-Za-z0-9]', '_', name)

    @staticmethod
    def _cache_tokenizer(real_model_name: str, cache_path: Union[str, Path]):
        tokenizer = AutoTokenizer.from_pretrained(real_model_name)
        tokenizer.save_pretrained(str(cache_path))

    def cache_model(self, model_name: str):
        if model_name in CLASSIFIERS:
            self.cache_classifier(model_name)
        elif model_name in EMBEDDERS:
            self.cache_embedding_model(model_name)
        else:
            raise KeyError(f'Unknown model: {model_name}')

    def cache_embedding_model(self, model_name: str):
        print(f'Checking for {model_name} in {self.cache_dir}')
        if not self.is_cached(model_name):
            real_model_name = EMBEDDERS[model_name].hf_name
            model_cache_path = str(self.get_model_path(model_name))
            print(f'Downloading and caching {model_name} ({real_model_name} at {model_cache_path})')
            os.makedirs(model_cache_path, exist_ok=True)
            if EMBEDDERS[model_name].kind == 'transformer':
                pretrained_model = SentenceTransformer(real_model_name)
                pretrained_model.save(model_cache_path)
            else:
                pretrained_model = AutoModel.from_pretrained(real_model_name)
                pretrained_model.save_pretrained(model_cache_path)
                self._cache_tokenizer(real_model_name, model_cache_path)
        else:
            print(f'Already cached {model_name}')

    def cache_classifier(self, model_name: str):
        print(f'Checking for {model_name} in {self.cache_dir}')
        if not self.is_cached(model_name):
            real_model_name = CLASSIFIERS[model_name].hf_name
            model_cache_path = str(self.get_model_path(model_name))
            print(f'Downloading and caching {model_name} ({real_model_name} at {model_cache_path})')
            os.makedirs(model_cache_path, exist_ok=True)
            pretrained_model = (
                AutoModelForSequenceClassification.from_pretrained(real_model_name)
            )
            pretrained_model.save_pretrained(model_cache_path)
            self._cache_tokenizer(real_model_name, model_cache_path)
        else:
            print(f'Already cached {model_name}')

    def cache_all_classifiers(self):
        for model_name in CLASSIFIERS.keys():
            self.cache_classifier(model_name)

    def cache_all_embeddings(self):
        for model_name in EMBEDDERS.keys():
            self.cache_embedding_model(model_name)

    def cache_all_models(self):
        self.cache_all_classifiers()
        self.cache_all_embeddings()

    @staticmethod
    def _set_cuda_settings():
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            torch.device("cuda")
            use_cuda = True
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print(f'We will use GPU {torch.cuda.current_device()}: '
                  f'{torch.cuda.get_device_name(torch.cuda.current_device())}')
        else:
            print('No GPU available, using the CPU instead.')
            torch.device("cpu")
            use_cuda = False
        return use_cuda

    def get_classifier(self, model_name: str):
        # ensure the model is downloaded
        self.cache_classifier(model_name)

        labels = CLASSIFIERS[model_name].labels
        model_cache_path = self.get_model_path(model_name)
        use_cuda = self._set_cuda_settings()
        device = 0 if use_cuda else -1

        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            model_cache_path,
            num_labels=len(labels),
            label2id={k: i for i, k in enumerate(labels)},
            id2label={i: k for i, k in enumerate(labels)})
        tokenizer = AutoTokenizer.from_pretrained(model_cache_path, use_fast=True)
        classifier = TextClassificationPipeline(
            model=pretrained_model, tokenizer=tokenizer, device=device)
        return classifier

    def get_embedder(self, model_name: str):
        self.cache_embedding_model(model_name)
        model_cache_path = str(self.get_model_path(model_name))
        if EMBEDDERS[model_name].kind == 'transformer':
            return SentenceTransformerBackend(model_cache_path, model_name)
        else:
            return AutoModelBackend(model_cache_path, model_name)
