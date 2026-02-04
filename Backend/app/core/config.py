from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os


class Settings(BaseSettings):
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    
    # Application Settings
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    # File Storage
    upload_dir: str = Field(default="./data/uploads", alias="UPLOAD_DIR")
    chroma_persist_dir: str = Field(default="./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    
    # CORS Settings (stored raw from env as comma-separated string)
    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        alias="CORS_ORIGINS"
    )

    @property
    def cors_origins(self) -> List[str]:
        """Return `CORS_ORIGINS` as a list parsed from comma-separated env value."""
        return [s.strip() for s in self.cors_origins_raw.split(",") if s.strip()]
    
    # OpenAI Models
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    chat_model: str = Field(default="gpt-4o-mini", alias="CHAT_MODEL")
    temperature: float = Field(default=0.2, alias="TEMPERATURE")
    
    # RAG Configuration
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    max_chunks_retrieval: int = Field(default=4, alias="MAX_CHUNKS_RETRIEVAL")
    
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
    if not settings.openai_api_key.startswith("sk-"):
        raise ValueError("Invalid OpenAI API key format. Must start with 'sk-'")
    
    if settings.chunk_size < settings.chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")
    
    print("✅ Configuration validated successfully")
    print(f"📍 Environment: {settings.environment}")
    print(f"🤖 Chat Model: {settings.chat_model}")
    print(f"📊 Embedding Model: {settings.embedding_model}")


# Run validation when module is imported
if __name__ != "__main__":
    validate_settings()