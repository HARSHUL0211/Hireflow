import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), case_sensitive=False, extra="ignore")

    # --- Google Gemini ---
    GOOGLE_API_KEY: str = Field(default="")
    GEMINI_API_KEY: str = Field(default="")  # accepted as an alias
    LLM_MODEL: str = Field(default="gemini-3.8-flash")
    LLM_TEMPERATURE: float = Field(default=0.2)
    EMBEDDING_MODEL: str = Field(default="models/gemini-embedding-001")
    EMBEDDING_DIMENSION: int = Field(default=768)

    # --- FAISS vector store (replaces Pinecone) ---
    FAISS_METRIC: str = Field(default="cosine")  # cosine == inner product on normalised vectors

    # --- Search / chunking ---
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)
    BM25_WEIGHT: float = Field(default=0.6)
    VECTOR_WEIGHT: float = Field(default=0.4)
    RRF_K: int = Field(default=60)
    MAX_TEXT_LENGTH: int = Field(default=4000)

    # --- Behaviour ---
    USE_LLM_PARSING: bool = Field(default=False)


config = AppConfig()

# Module-level constants (import these anywhere)
GOOGLE_API_KEY = config.GOOGLE_API_KEY or config.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL = config.LLM_MODEL
LLM_TEMPERATURE = config.LLM_TEMPERATURE
EMBEDDING_MODEL = config.EMBEDDING_MODEL
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION
FAISS_METRIC = config.FAISS_METRIC
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
BM25_WEIGHT = config.BM25_WEIGHT
VECTOR_WEIGHT = config.VECTOR_WEIGHT
RRF_K = config.RRF_K
MAX_TEXT_LENGTH = config.MAX_TEXT_LENGTH
USE_LLM_PARSING = config.USE_LLM_PARSING

# Paths
DATA_DIR = BASE_DIR / "data"
RESUMES_DIR = DATA_DIR / "resumes"
HYBRID_INDEX_DIR = DATA_DIR / "hybrid_index"
MEMORY_DIR = DATA_DIR / "memory"
FAISS_DIR = DATA_DIR / "faiss_index"
CACHE_DIR = DATA_DIR / "cache"
for _d in (RESUMES_DIR, HYBRID_INDEX_DIR, MEMORY_DIR, FAISS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)