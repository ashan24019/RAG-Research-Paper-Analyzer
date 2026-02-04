from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Settings(BaseSettings):
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Application Settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # File Storage
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    chroma_persist_dir: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIR")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # CORS Settings (stored raw from env as comma-separated string)
    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        env="CORS_ORIGINS"
    )

    @property
    def cors_origins(self) -> List[str]:
        """Return `CORS_ORIGINS` as a list parsed from comma-separated env value."""
        return [s.strip() for s in self.cors_origins_raw.split(",") if s.strip()]
    
    # OpenAI Models
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    chat_model: str = Field(default="gpt-4o-mini", env="CHAT_MODEL")
    temperature: float = Field(default=0.2, env="TEMPERATURE")
    
    # RAG Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_chunks_retrieval: int = Field(default=4, env="MAX_CHUNKS_RETRIEVAL")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow extra environment variables not declared in settings
        extra="allow",
        # Parse "null" strings as None
        env_parse_none_str="null",
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories on startup"""
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.chroma_persist_dir, exist_ok=True)
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment.lower() == "production"


# Create a global settings instance
settings = Settings()


# Validation function
def validate_settings():
    logger = logging.getLogger(__name__)

    if not settings.openai_api_key or not settings.openai_api_key.strip():
        raise ValueError("Missing OPENAI_API_KEY environment variable")

    if settings.chunk_size < settings.chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    logger.info("Configuration validated successfully")
    logger.info("Environment: %s", settings.environment)
    logger.info("Chat Model: %s", settings.chat_model)
    logger.info("Embedding Model: %s", settings.embedding_model)