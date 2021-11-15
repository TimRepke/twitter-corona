import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer
from abc import ABC, abstractmethod
# most of this code is taken from
# https://github.com/MaartenGr/BERTopic/tree/15ea0cd804d35c1f11c6692f33c3666b648dd6c8/bertopic/backend


class BaseEmbedder(ABC):
    """ The Base Embedder used for creating embedding models
    Arguments:
        embedding_model: The main embedding model to be used for extracting
                         document and word embedding
    """

    def __init__(self, embedding_model:str=None):
        self.embedding_model = embedding_model

    @abstractmethod
    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        """ Embed a list of n documents/words into an n-dimensional
        matrix of embeddings
        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process
        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        raise NotImplementedError()

    def embed_words(self,
                    words: List[str],
                    verbose: bool = False) -> np.ndarray:
        """ Embed a list of n words into an n-dimensional
        matrix of embeddings
        Arguments:
            words: A list of words to be embedded
            verbose: Controls the verbosity of the process
        Returns:
            Word embeddings with shape (n, m) with `n` words
            that each have an embeddings size of `m`
        """
        return self.embed(words, verbose)

    def embed_documents(self,
                        documents: List[str],
                        verbose: bool = False) -> np.ndarray:
        """ Embed a list of n words into an n-dimensional
        matrix of embeddings
        Arguments:
            documents: A list of documents to be embedded
            verbose: Controls the verbosity of the process
        Returns:
            Document embeddings with shape (n, m) with `n` documents
            that each have an embeddings size of `m`
        """
        return self.embed(documents, verbose)


# class AutoModelBackend(BaseEmbedder):
#
#     def __init__(self, embedding_model: str):
#         super().__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained(embedding_model, use_fast=False)
#         self.embedding_model = AutoModel.from_pretrained(embedding_model)
#
#     def embed(self,
#               documents: List[str],
#               verbose: bool = False) -> np.ndarray:
#
#         tokenised = torch.tensor([self.tokenizer.encode(d) for d in documents])
#         embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
#         return embeddings



class SentenceTransformerBackend(BaseEmbedder):
    """ Sentence-transformers embedding model
    The sentence-transformers embedding model used for generating document and
    word embeddings.
    Arguments:
        embedding_model: A sentence-transformers embedding model
    Usage:
    To create a model, you can load in a string pointing to a
    sentence-transformers model:
    ```python
    from bertopic.backend import SentenceTransformerBackend
    sentence_model = SentenceTransformerBackend("all-MiniLM-L6-v2")
    ```
    or  you can instantiate a model yourself:
    ```python
    from bertopic.backend import SentenceTransformerBackend
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_model = SentenceTransformerBackend(embedding_model)
    ```
    """

    def __init__(self, embedding_model: str):
        super().__init__()
        self.embedding_model = SentenceTransformer(embedding_model)

    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        """ Embed a list of n documents/words into an n-dimensional
        matrix of embeddings
        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process
        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings



