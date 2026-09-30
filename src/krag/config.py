"""Application configuration via environment variables (pydantic-settings)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = Field(default="local")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    graph_backend: str = Field(default="memory")  # "real" (Neo4j) | "memory"
    vector_backend: str = Field(default="memory")  # "real" (pgvector) | "memory"

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="changeme")

    database_url: str = Field(default="postgresql://krag:krag@localhost:5432/krag")

    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    # OpenAI-compatible LLM provider. Empty -> deterministic stub provider.
    llm_base_url: str = Field(default="")
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="")
    llm_timeout_seconds: int = Field(default=60)

    qa_top_k_vector: int = Field(default=8)
    qa_top_k_graph: int = Field(default=12)
    qa_max_hops: int = Field(default=3)
    router_low_confidence_threshold: float = Field(default=0.55)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Test hook: force re-read of env vars."""
    global _settings
    _settings = None
