import numpy as np
from scipy.sparse.csr import csr_matrix
from typing import Type, Optional, Union, Literal
from dataclasses import dataclass, asdict

# from sklearn.cluster import SpectralClustering # don't use this, memory explodes
# from spectralcluster import SpectralClusterer # don't use this, memory explodes
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from utils.tweets import Tweets
from utils.embedding import SentenceTransformerBackend, BaseEmbedder


@dataclass
class VectorizerArgs:
    min_df: Union[float, int] = 0
    max_df: Union[float, int] = 1.0
    ngram_range: tuple[int, int] = (1, 1)
    max_features: int = None
    lowercase: bool = True
    use_idf: bool = True
    smooth_idf: bool = True
    stop_words: Optional[Union[set[str], list[str], Literal['english']]] = None


@dataclass
class UMAPArgs:
    n_neighbors: int = 15
    n_components: int = 2
    metric: str = "cosine"
    output_metric: str = "euclidean"
    min_dist: float = 0.1
    spread: float = 1.0
    local_connectivity: float = 1.0
    repulsion_strength: float = 1.0
    negative_sample_rate: int = 5
    random_state: bool = None
    densmap: bool = False
    dens_lambda: float = 2.0
    dens_frac: float = 0.3
    dens_var_shift: float = 0.1


def mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: list[str],
        top_n: int = 5,
        diversity: float = 0.8) -> list[str]:
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr_ = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr_)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]


class FrankenTopic:
    def __init__(self,
                 n_topics: int = 20,
                 n_words_per_topic: int = 20,
                 n_candidates: int = 40,
                 mmr_diversity: float = 0.8,
                 emb_backend: Type[BaseEmbedder] = SentenceTransformerBackend,
                 emb_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 umap_args: UMAPArgs = None,
                 vectorizer_args: VectorizerArgs = None):
        if vectorizer_args is None:
            vectorizer_args = VectorizerArgs()
        if umap_args is None:
            umap_args = UMAPArgs()

        self.vectorizer_args = vectorizer_args
        self.umap_args = umap_args
        self.n_topics = n_topics
        self.n_candidates = n_candidates
        self.n_words_per_topic = n_words_per_topic
        self.mmr_diversity = mmr_diversity

        self.emb_backend = emb_backend
        self.emb_model = emb_model

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.vocab: Optional[dict[int, str]] = None
        self.tf_idf_vecs: Optional[csr_matrix] = None
        self.umap: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self._is_fit = False

    def fit(self, tweets: Tweets, embeddings: np.ndarray):
        print('Fitting UMAP...')
        mapper = UMAP(**asdict(self.umap_args))
        self.umap = mapper.fit_transform(embeddings)

        print('Clustering...')
        # clustering = SpectralClustering(n_clusters=self.n_topics, n_jobs=2)
        # self.labels = clustering.fit_predict(self.umap)
        # clusterer = SpectralClusterer(min_clusters=15, max_clusters=25,
        #                               custom_dist='euclidean')
        # self.labels = clusterer.predict(self.umap)
        clustering = KMeans(n_clusters=self.n_topics)
        self.labels = clustering.fit_predict(self.umap)

        print('Grouping tweets...')
        grouped_texts = [
            [tweets.tweets[i].clean_text for i in np.argwhere(self.labels == label).reshape(-1, )]
            for label in np.unique(self.labels)
        ]

        # note, that BERTopic does something slightly different:
        # https://github.com/MaartenGr/BERTopic/blob/15ea0cd804d35c1f11c6692f33c3666b648dd6c8/bertopic/_ctfidf.py
        print('Vectorising groups...')
        self.vectorizer = TfidfVectorizer(**asdict(self.vectorizer_args))
        self.tf_idf_vecs = self.vectorizer.fit_transform([' '.join(g) for g in grouped_texts])
        self.vocab = {v: k for k, v in self.vectorizer.vocabulary_.items()}

        self._is_fit = True

    def get_top_n_mmr(self, n_tokens: int = None) -> list[list[tuple[str, float]]]:
        assert self._is_fit

        if n_tokens is None:
            n_tokens = self.n_words_per_topic

        topics_tfidf = self.get_top_n_tfidf(self.n_candidates)
        topics_mmr = []
        embedder = self.emb_backend(self.emb_model)
        print('Improving topics keywords...')
        for topic in topics_tfidf:
            words = [w[0] for w in topic]
            word_embeddings = embedder.embed_words(words, verbose=False)
            topic_embedding = embedder.embed_documents([' '.join(words)], verbose=False).reshape(1, -1)
            topic_words = mmr(topic_embedding, word_embeddings, words,
                              top_n=n_tokens, diversity=self.mmr_diversity)
            topics_mmr.append([
                (word, value) for word, value in topic if word in topic_words
            ])
        return topics_mmr

    def get_top_n_tfidf(self, n_tokens: int = 20) -> list[list[tuple[str, float]]]:
        print('Computing top tf-idf words per topic...')
        rank = np.argsort(self.tf_idf_vecs.todense())
        return [
            [(self.vocab[rank[i, -(j + 1)]], self.tf_idf_vecs[i, rank[i, -(j + 1)]])
             for j in range(n_tokens)]
            for i in range(len(rank))
        ]
