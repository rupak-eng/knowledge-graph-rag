"""Vector store: protocol + pgvector backend (primary) + numpy backend (tests).

Chunk IDs are shared with the graph store — the same chunk_id written to
Neo4j is stored as metadata in pgvector, enabling cross-store joins.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from krag.domain.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int: ...
    def search(
        self, query_embedding: list[float], top_k: int = 8
    ) -> list[RetrievedChunk]: ...
    def get(self, chunk_id: str) -> RetrievedChunk | None: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...
    def close(self) -> None: ...


class InMemoryVectorStore:
    """Brute-force cosine search. Deterministic; for tests and offline eval."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vecs: dict[str, np.ndarray] = {}

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        added = 0
        for chunk, emb in zip(chunks, embeddings):
            vec = np.asarray(emb, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            self._chunks[chunk.chunk_id] = chunk
            self._vecs[chunk.chunk_id] = vec
            added += 1
        return added

    def search(
        self, query_embedding: list[float], top_k: int = 8
    ) -> list[RetrievedChunk]:
        if not self._vecs:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        scored = [(float(q @ v), cid) for cid, v in self._vecs.items()]
        scored.sort(reverse=True)
        out: list[RetrievedChunk] = []
        for score, cid in scored[:top_k]:
            c = self._chunks[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    section=c.section,
                    text=c.text,
                    score=score,
                    source="vector",
                )
            )
        return out

    def get(self, chunk_id: str) -> RetrievedChunk | None:
        c = self._chunks.get(chunk_id)
        if c is None:
            return None
        return RetrievedChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            section=c.section,
            text=c.text,
            score=1.0,
            source="vector",
        )

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vecs.clear()

    def close(self) -> None:
        pass


class PgVectorStore:
    """Production backend: Postgres + pgvector with HNSW index."""

    def __init__(self, database_url: str, dim: int = EMBEDDING_DIM) -> None:
        import psycopg

        self._dim = dim
        self._conn = psycopg.connect(database_url)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                section TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                embedding vector(%s) NOT NULL
            )
            """,
            (dim,),
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
            """
        )
        self._conn.commit()
        logger.info("Connected to pgvector (dim=%d)", dim)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        from pgvector.psycopg import Vector

        with self._conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO chunks (chunk_id, doc_id, section, text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        section = EXCLUDED.section,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.section,
                        chunk.text,
                        Vector(emb),
                    ),
                )
        self._conn.commit()
        return len(chunks)

    def search(
        self, query_embedding: list[float], top_k: int = 8
    ) -> list[RetrievedChunk]:
        from pgvector.psycopg import Vector

        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_id, section, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (Vector(query_embedding), Vector(query_embedding), top_k),
            )
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                chunk_id=r[0],
                doc_id=r[1],
                section=r[2],
                text=r[3],
                score=float(r[4]),
                source="vector",
            )
            for r in rows
        ]

    def get(self, chunk_id: str) -> RetrievedChunk | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, doc_id, section, text FROM chunks WHERE chunk_id = %s",
                (chunk_id,),
            )
            r = cur.fetchone()
        if r is None:
            return None
        return RetrievedChunk(
            chunk_id=r[0], doc_id=r[1], section=r[2], text=r[3], score=1.0, source="vector"
        )

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def build_vector_store(database_url: str, dim: int, backend: str) -> VectorStore:
    if backend == "real":
        return PgVectorStore(database_url, dim)
    if backend == "memory":
        return InMemoryVectorStore()
    raise ValueError(f"unknown VECTOR_BACKEND: {backend}")
