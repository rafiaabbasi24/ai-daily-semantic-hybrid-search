import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HybridSearchEngine:
    """
    Hybrid search engine that combines dense semantic embeddings (via
    SentenceTransformers) with sparse TF‑IDF vectors (via scikit‑learn).
    The two similarity signals are linearly blended using a configurable
    ``alpha`` weight:

    ``final_score = alpha * dense_score + (1 - alpha) * sparse_score``

    The engine is language‑agnostic as long as the underlying SentenceTransformer
    model supports the target language(s).

    Example
    -------
    >>> engine = HybridSearchEngine()
    >>> docs = [
    ...     {"id": "1", "text": "The cat sits on the mat."},
    ...     {"id": "2", "text": "A dog barked loudly."}
    ... ]
    >>> engine.add_documents(docs)
    >>> results = engine.search("feline on carpet", top_k=1)
    >>> results[0]["id"]
    '1'
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        tfidf_max_features: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialise the hybrid engine.

        Parameters
        ----------
        model_name: str
            Hugging‑Face identifier of the SentenceTransformer model.
        tfidf_max_features: int | None
            Maximum number of features for the TF‑IDF vectoriser. ``None`` means
            unlimited.
        device: str | None
            Torch device (e.g. ``"cpu"``, ``"cuda"``). If ``None``, automatically
            selects ``cuda`` when available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading SentenceTransformer model %s on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            stop_words="english",
            norm="l2",
        )

        # Internal stores
        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._dense_embeddings: Optional[np.ndarray] = None  # shape (n_docs, dim)
        self._sparse_matrix: Optional[sparse.csr_matrix] = None  # TF‑IDF matrix

    # --------------------------------------------------------------------- #
    # Document handling
    # --------------------------------------------------------------------- #
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add a batch of documents to the index. Existing documents are cleared
        and the whole index is rebuilt.

        Each document must contain at least two keys:
        ``"id"`` – a unique identifier (string) and ``"text"`` – the raw content.

        Parameters
        ----------
        documents: List[Dict[str, Any]]
            List of document dictionaries.
        """
        logger.info("Adding %d documents to the index", len(documents))
        self._doc_ids = [str(doc["id"]) for doc in documents]
        self._doc_texts = [str(doc["text"]) for doc in documents]

        # Compute dense embeddings and sparse TF‑IDF in one pass
        self._dense_embeddings = self._compute_dense_embeddings(self._doc_texts)
        self._sparse_matrix = self._compute_sparse_matrix(self._doc_texts)

        logger.info(
            "Index built: %d docs, dense dim=%d, sparse features=%d",
            len(self._doc_ids),
            self._dense_embeddings.shape[1],
            self._sparse_matrix.shape[1],
        )

    # --------------------------------------------------------------------- #
    # Embedding / TF‑IDF computation
    # --------------------------------------------------------------------- #
    def _compute_dense_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute L2‑normalised dense embeddings for a list of texts.

        Returns
        -------
        np.ndarray
            2‑D array of shape (len(texts), embedding_dim) with dtype ``float32``.
        """
        logger.debug("Computing dense embeddings for %d texts", len(texts))
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )
        # Ensure float32 for downstream cosine similarity
        return embeddings.astype(np.float32)

    def _compute_sparse_matrix(self, texts: List[str]) -> sparse.csr_matrix:
        """
        Fit the TF‑IDF vectoriser on the provided texts and transform them
        into a sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape (len(texts), n_features) with L2‑normalised rows.
        """
        logger.debug("Fitting TF‑IDF vectoriser on %d texts", len(texts))
        tfidf = self.tfidf_vectorizer.fit_transform(texts)
        # Normalisation is already performed by the vectoriser (`norm='l2'`),
        # but we re‑normalise to guard against edge cases (e.g. empty docs).
        tfidf = sparse.linalg.norm(tfidf, axis=1, keepdims=True)
        tfidf = tfidf.multiply(1 / (tfidf + 1e-10))
        return tfidf

    # --------------------------------------------------------------------- #
    # Search
    # --------------------------------------------------------------------- #
    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search for ``query`` and return the top‑k results.

        Parameters
        ----------
        query: str
            The user query.
        top_k: int
            Number of results to return.
        alpha: float
            Weight for the dense component (0 ≤ alpha ≤ 1). ``1`` uses only dense,
            ``0`` uses only sparse.

        Returns
        -------
        List[Dict[str, Any]]
            Each entry contains ``id``, ``text`` and ``score`` (higher is better).
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1")

        if not self._doc_ids:
            logger.warning("Search called on empty index")
            return []

        logger.debug("Encoding query for dense search")
        query_dense = self._compute_dense_embeddings([query])[0]  # shape (dim,)

        logger.debug("Encoding query for sparse search")
        query_sparse = self.tfidf_vectorizer.transform([query])  # shape (1, n_features)

        # Dense similarity: cosine (since vectors are L2‑normalised)
        dense_scores = np.dot(self._dense_embeddings, query_dense)  # (n_docs,)

        # Sparse similarity: dot product (vectors already L2‑normalised)
        sparse_scores = query_sparse @ self._sparse_matrix.T  # (1, n_docs)
        sparse_scores = np.asarray(sparse_scores).ravel()

        # Blend the scores
        final_scores = alpha * dense_scores + (1.0 - alpha) * sparse_scores

        # Get top‑k indices
        top_indices = np.argpartition(-final_scores, range(min(top_k, len(final_scores))))[
            :top_k
        ]
        # Sort the top indices by score descending
        top_indices = top_indices[np.argsort(-final_scores[top_indices])]

        results = [
            {
                "id": self._doc_ids[idx],
                "text": self._doc_texts[idx],
                "score": float(final_scores[idx]),
            }
            for idx in top_indices
        ]

        logger.debug(
            "Search completed for query='%s' (top_k=%d, alpha=%.2f)", query, top_k, alpha
        )
        return results

    # --------------------------------------------------------------------- #
    # Utility helpers
    # --------------------------------------------------------------------- #
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its identifier.

        Parameters
        ----------
        doc_id: str
            Identifier of the document.

        Returns
        -------
        dict | None
            Document dict with ``id`` and ``text`` if found, else ``None``.
        """
        try:
            idx = self._doc_ids.index(str(doc_id))
        except ValueError:
            return None
        return {"id": self._doc_ids[idx], "text": self._doc_texts[idx]}

    def clear(self) -> None:
        """
        Remove all indexed documents and reset internal state.
        """
        logger.info("Clearing the entire index")
        self._doc_ids.clear()
        self._doc_texts.clear()
        self._dense_embeddings = None
        self._sparse_matrix = None
        # Reset TF‑IDF vectoriser without discarding the fitted state (scikit‑learn
        # does not provide an official reset, so we re‑instantiate).
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_vectorizer.max_features,
            stop_words="english",
            norm="l2",
        )