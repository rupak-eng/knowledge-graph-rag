"""Embedding provider abstraction: sentence-transformers (real) / stub (tests)."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerProvider:
    """Real embeddings via sentence-transformers (runs offline after download)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim: int = int(self._model.get_sentence_embedding_dimension())
        logger.info("Loaded embedding model %s (dim=%d)", model_name, self.dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        vecs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        logger.debug("embedded %d texts in %.2fs", len(texts), time.perf_counter() - t0)
        return [list(map(float, v)) for v in vecs]


class StubEmbeddingProvider:
    """Deterministic hash-based embeddings for unit tests (no downloads)."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            rng = np.random.default_rng(
                int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
            )
            vec = rng.standard_normal(self.dim).astype(np.float32)
            vec /= float(np.linalg.norm(vec)) + 1e-12
            out.append([float(x) for x in vec])
        return out
