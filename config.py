"""
config.py - Loads all settings from .env using python-dotenv
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "flights")

    # Models
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    COHERE_EMBED_MODEL: str = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
    COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")

    # Server
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", 8000))

    # RAG tuning
    TOP_K_RETRIEVE: int = 10   # Candidates fetched from ChromaDB
    TOP_K_RERANK: int = 3      # Kept after Cohere reranking
    MEMORY_WINDOW: int = 6     # Past messages to keep per session

    def validate(self):
        missing = []
        if not self.COHERE_API_KEY or "your_" in self.COHERE_API_KEY:
            missing.append("COHERE_API_KEY")
        if not self.GROQ_API_KEY or "your_" in self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required API keys in .env: {', '.join(missing)}"
            )


cfg = Config()
