import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

# most of this code is taken from
# https://github.com/MaartenGr/BERTopic/tree/15ea0cd804d35c1f11c6692f33c3666b648dd6c8/bertopic/backend


class BaseEmbedder:
    """ The Base Embedder used for creating embedding models
    Arguments:
        embedding_model: The main embedding model to be used for extracting
                         document and word embedding
        word_embedding_model: The embedding model used for extracting word
                              embeddings only. If this model is selected,
                              then the `embedding_model` is purely used for
                              creating document embeddings.
    """

    def __init__(self,
                 embedding_model=None,
                 word_embedding_model=None):
        self.embedding_model = embedding_model
        self.word_embedding_model = word_embedding_model

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
        pass

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
                        document: List[str],
                        verbose: bool = False) -> np.ndarray:
        """ Embed a list of n words into an n-dimensional
        matrix of embeddings
        Arguments:
            document: A list of documents to be embedded
            verbose: Controls the verbosity of the process
        Returns:
            Document embeddings with shape (n, m) with `n` documents
            that each have an embeddings size of `m`
        """
        return self.embed(document, verbose)


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

    def __init__(self, embedding_model: Union[str, SentenceTransformer]):
        super().__init__()

        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            raise ValueError("Please select a correct SentenceTransformers model: \n"
                             "`from sentence_transformers import SentenceTransformer` \n"
                             "`model = SentenceTransformer('all-MiniLM-L6-v2')`")

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



