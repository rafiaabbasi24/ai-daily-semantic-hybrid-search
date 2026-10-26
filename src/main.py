import os
import uuid
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy import sparse

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Environment variables can be used to override defaults in production
DEFAULT_DENSE_MODEL = os.getenv(
    "DENSE_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_HYBRID_WEIGHT = float(os.getenv("HYBRID_WEIGHT", "0.5"))  # 0.0 -> only sparse, 1.0 -> only dense

# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


class Document(BaseModel):
    """Schema for a single document to be indexed."""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the document. Generated if omitted.",
    )
    text: str = Field(..., description="Raw text content of the document.")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional free‑form metadata."
    )

    @validator("text")
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Document text must be non‑empty")
        return v


class SearchRequest(BaseModel):
    """Payload for a search query."""

    query: str = Field(..., description="User query string.")
    top_k: Optional[int] = Field(
        None,
        description="Number of results to return. Falls back to server default if omitted.",
    )
    hybrid_weight: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Weight for dense component in hybrid scoring (0 = only sparse, 1 = only dense).",
    )


class SearchResult(BaseModel):
    """Single search result."""

    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class SearchResponse(BaseModel):
    """Response payload for a search request."""

    query: str
    results: List[SearchResult]


# --------------------------------------------------------------------------- #
# Core indexing and retrieval logic
# --------------------------------------------------------------------------- #


class HybridSemanticSearchEngine:
    """
    In‑memory hybrid search engine that combines dense embeddings from a
    SentenceTransformer model with sparse TF‑IDF vectors.
    """

    def __init__(
        self,
        dense_model_name: str = DEFAULT_DENSE_MODEL,
        sparse_max_features: int = 50000,
    ) -> None:
        # Load dense model
        self.dense_model = SentenceTransformer(dense_model_name)

        # Initialise sparse vectoriser – using word‑level tokenisation works well for many languages.
        self.sparse_vectoriser = TfidfVectorizer(
            max_features=sparse_max_features,
            stop_words=None,
            norm="l2",
        )

        # Internal storage
        self.documents: List[Document] = []
        self._dense_embeddings: Optional[np.ndarray] = None  # shape (n_docs, dim)
        self._sparse_matrix: Optional[sparse.csr_matrix] = None  # shape (n_docs, vocab)

    # --------------------------------------------------------------------- #
    # Document management
    # --------------------------------------------------------------------- #

    def add_documents(self, docs: List[Document]) -> None:
        """
        Add new documents to the index. This method recomputes both dense and
        sparse representations for the entire collection for simplicity.
        """
        # Append new docs
        self.documents.extend(docs)

        # Re‑compute dense embeddings
        texts = [doc.text for doc in self.documents]
        dense_emb = self.dense_model.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )
        self._dense_embeddings = dense_emb  # (n_docs, dim)

        # Re‑compute sparse matrix
        # Fit on the full corpus each time; for large corpora a more sophisticated
        # incremental approach would be required.
        self._sparse_matrix = self.sparse_vectoriser.fit_transform(texts)

    # --------------------------------------------------------------------- #
    # Retrieval
    # --------------------------------------------------------------------- #

    def _ensure_indexed(self) -> None:
        """Raise an informative error if the engine has no indexed documents."""
        if not self.documents:
            raise HTTPException(status_code=404, detail="No documents have been indexed yet.")
        if self._dense_embeddings is None or self._sparse_matrix is None:
            raise HTTPException(
                status_code=500,
                detail="Indexing state is corrupted – embeddings or sparse matrix missing.",
            )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        hybrid_weight: float = DEFAULT_HYBRID_WEIGHT,
    ) -> List[SearchResult]:
        """
        Perform a hybrid search.

        Parameters
        ----------
        query: str
            The user query.
        top_k: int
            Number of results to return.
        hybrid_weight: float
            Weight for the dense component (0 → only sparse, 1 → only dense).

        Returns
        -------
        List[SearchResult]
            Ranked results with scores in the range [0, 1].
        """
        self._ensure_indexed()

        # ---- Dense similarity -------------------------------------------------
        query_dense = self.dense_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )  # shape (1, dim)
        dense_scores = cosine_similarity(query_dense, self._dense_embeddings).flatten()
        # dense_scores already in [0, 1] because of L2 normalisation

        # ---- Sparse similarity ------------------------------------------------
        query_sparse = self.sparse_vectoriser.transform([query])  # shape (1, vocab)
        # linear_kernel returns the dot product; because TF‑IDF vectors are L2‑normalised,
        # the result is also a cosine similarity.
        sparse_scores = linear_kernel(query_sparse, self._sparse_matrix).flatten()

        # ---- Hybrid combination -----------------------------------------------
        # Normalise each score vector to sum to 1 (optional but stabilises weighting)
        dense_norm = dense_scores / (dense_scores.sum() + 1e-12)
        sparse_norm = sparse_scores / (sparse_scores.sum() + 1e-12)

        combined = hybrid_weight * dense_norm + (1.0 - hybrid_weight) * sparse_norm

        # Get top‑k indices
        top_indices = np.argpartition(-combined, range(min(top_k, len(combined))))[:top_k]
        top_indices = top_indices[np.argsort(-combined[top_indices])]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    id=doc.id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=float(combined[idx]),
                )
            )
        return results


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Multilingual Hybrid Semantic Search API",
    description=(
        "Provides dense‑plus‑sparse (hybrid) semantic search over multilingual text. "
        "Documents are stored in‑memory for demonstration purposes."
    ),
    version="1.0.0",
)

# Enable CORS for all origins – adjust in production as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise the search engine singleton
search_engine = HybridSemanticSearchEngine()


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #


@app.post(
    "/documents",
    summary="Add documents to the index",
    response_model=Dict[str, Any],
    status_code=201,
)
def add_documents(docs: List[Document] = Body(..., embed=True)):
    """
    Index a batch of documents. The whole corpus is re‑encoded on each call,
    which is acceptable for small‑scale demos. For production, consider an
    incremental indexing strategy.
    """
    if not docs:
        raise HTTPException(status_code=400, detail="Document list cannot be empty.")
    search_engine.add_documents(docs)
    return {"message": f"Indexed {len(docs)} document(s).", "total_documents": len(search_engine.documents)}


@app.post(
    "/search",
    summary="Hybrid semantic search",
    response_model=SearchResponse,
)
def search_endpoint(payload: SearchRequest):
    """
    Perform a hybrid search over the indexed documents. Returns the top‑k
    most relevant results together with a relevance score.
    """
    top_k = payload.top_k or DEFAULT_TOP_K
    hybrid_weight = (
        payload.hybrid_weight
        if payload.hybrid_weight is not None
        else DEFAULT_HYBRID_WEIGHT
    )
    results = search_engine.search(query=payload.query, top_k=top_k, hybrid_weight=hybrid_weight)
    return SearchResponse(query=payload.query, results=results)


@app.get(
    "/healthz",
    summary="Health check endpoint",
    response_model=Dict[str, str],
)
def health_check():
    """Simple health‑check used by orchestrators."""
    return {"status": "ok"}

# --------------------------------------------------------------------------- #
# If the module is executed directly, start a development server.
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)