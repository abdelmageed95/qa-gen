"""Application configuration settings"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # LandingAI Configuration
    VISION_AGENT_API_KEY: Optional[str] = None
    LANDINGAI_DPT_MODEL: str = "dpt-2-latest"
    USE_LANDINGAI_CHUNKING: bool = False  # Toggle between LandingAI and legacy chunking

    # ChromaDB Configuration
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # Document Processing Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    RETRIEVAL_TOP_K: int = 5
    MIN_RELEVANCE_THRESHOLD: float = 1.5  # Max cosine distance (0=identical, 2=opposite)

    # Agent Configuration
    MAX_ITERATIONS: int = 3
    MIN_QUESTION_SCORE: float = 0.7  # Minimum score for question approval

    # Application Configuration
    UPLOAD_DIR: str = "./uploads"
    DOCUMENT_REGISTRY_PATH: str = "./chroma_db/documents.json"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
