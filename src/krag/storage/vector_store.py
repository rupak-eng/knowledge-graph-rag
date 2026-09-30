"""Vector store: protocol + pgvector backend (primary) + numpy backend (tests).

Chunk IDs are shared with the graph store — the same chunk_id written to
Neo4j is stored as metadata in pgvector, enabling cross-store joins.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

import numpy as np

from krag.domain.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int: ...
    def search(self, query_embedding: list[float], top_k: int = 8) -> list[RetrievedChunk]: ...
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
        for chunk, emb in zip(chunks, embeddings, strict=True):
            vec = np.asarray(emb, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            self._chunks[chunk.chunk_id] = chunk
            self._vecs[chunk.chunk_id] = vec
            added += 1
        return added

    def search(self, query_embedding: list[float], top_k: int = 8) -> list[RetrievedChunk]:
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

    def __init__(self, database_url: str, dim: int = EMBEDDING_DIM, table: str = "chunks") -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        # DDL identifiers must be literals in Postgres; dim comes from our own
        # config (validated int) and table is restricted to a safe identifier
        # pattern — neither is user input.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError(f"unsafe table name: {table!r}")
        self._dim = int(dim)
        self._table = table
        self._conn = psycopg.connect(database_url)
        register_vector(self._conn)  # adapt plain float lists <-> vector
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Guard: an existing table with a different dimension is a hard
        # error (re-embedding is required); fail loudly instead of writing
        # silently truncated/padded vectors.
        existing = self._conn.execute(
            """
            SELECT atttypmod FROM pg_attribute
            JOIN pg_class ON pg_class.oid = pg_attribute.attrelid
            WHERE pg_class.relname = %s AND pg_attribute.attname = 'embedding'
            """,
            (table,),
        ).fetchone()
        if existing is not None and int(existing[0]) != self._dim:
            raise ValueError(
                f"{table}.embedding is vector({existing[0]}), "
                f"but store was opened with dim={self._dim}. "
                "Drop the table (data must be re-embedded) or use the matching dim."
            )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                section TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                embedding vector({self._dim}) NOT NULL
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
            ON {table} USING hnsw (embedding vector_cosine_ops)
            """
        )
        self._conn.commit()
        logger.info("Connected to pgvector (dim=%d, table=%s)", dim, table)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        t = self._table
        with self._conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings, strict=True):
                cur.execute(
                    f"""
                    INSERT INTO {t} (chunk_id, doc_id, section, text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        section = EXCLUDED.section,
                        embedding = EXCLUDED.embedding
                    """,
                    (chunk.chunk_id, chunk.doc_id, chunk.section, chunk.text, emb),
                )
        self._conn.commit()
        return len(chunks)

    def search(self, query_embedding: list[float], top_k: int = 8) -> list[RetrievedChunk]:
        t = self._table
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_id, doc_id, section, text,
                       1 - (embedding <=> %s::vector) AS score
                FROM {t}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
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
                f"SELECT chunk_id, doc_id, section, text FROM {self._table} WHERE chunk_id = %s",
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
            cur.execute(f"SELECT count(*) FROM {self._table}")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table}")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def build_vector_store(
    database_url: str, dim: int, backend: str, table: str = "chunks"
) -> VectorStore:
    if backend == "real":
        return PgVectorStore(database_url, dim, table=table)
    if backend == "memory":
        return InMemoryVectorStore()
    raise ValueError(f"unknown VECTOR_BACKEND: {backend}")
