from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o"

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "codebase-rag"

    # Cohere
    COHERE_API_KEY: str

    # Groq
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-70b-8192"

    # Retrieval settings
    DENSE_TOP_K: int = 15
    SPARSE_TOP_K: int = 15
    RERANK_TOP_N: int = 8
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    class Config:
        env_file = ".env"

@lru_cache()
def getSettings()-> Settings:
    return Settings()

