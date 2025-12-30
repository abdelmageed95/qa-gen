"""Dependency injection for FastAPI"""

from functools import lru_cache
from app.core.config import Settings

@lru_cache()# Cache settings instance
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
