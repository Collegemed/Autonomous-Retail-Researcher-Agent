"""
config.py - Centralized configuration using environment variables.
All API keys and settings are loaded here via python-dotenv.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file from project root

class Config:
    # ─── LLM ─────────────────────────────────────────────────────────────────
    # Supports: "groq", "openai", "anthropic"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

    # ─── Search ───────────────────────────────────────────────────────────────
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    SEARCH_TIMEOUT: int = int(os.getenv("SEARCH_TIMEOUT", "15"))

    # ─── Database ─────────────────────────────────────────────────────────────
    DB_PATH: str = os.getenv("DB_PATH", "retail_researcher.db")

    # ─── RAG / Embeddings ─────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))

    # ─── Performance ──────────────────────────────────────────────────────────
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    # ─── Token Limits ─────────────────────────────────────────────────────────
    MAX_INPUT_TOKENS: int = int(os.getenv("MAX_INPUT_TOKENS", "3000"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

config = Config()